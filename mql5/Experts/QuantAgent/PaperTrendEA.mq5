//+------------------------------------------------------------------+
//| PaperTrendEA.mq5                                                  |
//| 受治理的趋势跟踪参考 EA / governed trend-following reference EA.   |
//|                                                                   |
//| 这是一个**教学与验证**用 EA，不是策略建议。它存在的目的是示范       |
//| MQL5 的正确写法：指标句柄 + CopyBuffer、OnTradeTransaction 处理、   |
//| A 股手数与时段规则、以及默认拒绝实盘账户。                          |
//|                                                                   |
//| A reference EA for teaching and validation, not a strategy         |
//| recommendation. It exists to demonstrate correct MQL5: indicator   |
//| handles read via CopyBuffer, OnTradeTransaction handling, A-share  |
//| lot and session rules, and refusal to run on a real account.       |
//|                                                                   |
//| 明确不做的事 / deliberately absent:                                |
//|   * 马丁格尔、无限网格、加倍摊平 —— 尾部风险不可接受               |
//|   * 任何真实账户下单路径                                            |
//|   * 把测试器生成的 tick 当作真实 tick 报告                          |
//+------------------------------------------------------------------+
#property copyright "QuantAgent"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>
#include <QuantAgent/Logging.mqh>
#include <QuantAgent/AShareGuards.mqh>
#include <QuantAgent/IndicatorHandle.mqh>
#include <QuantAgent/RiskGuard.mqh>

//--- 输入参数 / inputs
input int      InpFastPeriod        = 10;      // 快线周期
input int      InpSlowPeriod        = 30;      // 慢线周期
input int      InpAtrPeriod         = 14;      // ATR 周期
input double   InpTargetShares      = 1000;    // 目标股数 (SHARES, not lots)
input int      InpMaxOrdersPerDay   = 10;      // 单日最大下单笔数
input double   InpMaxNotionalCNY    = 500000;  // 单日最大成交金额
input double   InpMaxDailyLossCNY   = 10000;   // 单日最大亏损
input int      InpMaxQuoteAgeSec    = 120;     // 行情最大陈旧秒数
input bool     InpDeclaredRealTicks = false;   // 运行方声明：测试器使用真实 tick
input bool     InpAllowRealAccount  = false;   // 必须保持 false

//--- 全局状态
CTrade         g_trade;
QAIndicator    g_fast, g_slow, g_atr;
QARiskGuard    g_risk;
ENUM_QA_BOARD  g_board = QA_BOARD_UNKNOWN;
bool           g_initialised = false;

//+------------------------------------------------------------------+
int OnInit()
  {
   QA_SetLogLevel(QA_LOG_INFO);

   //--- 硬性前置：实盘账户直接拒绝初始化
   if(!QA_AssertNotRealAccount(InpAllowRealAccount))
      return INIT_FAILED;

   //--- 只允许在本仓库导入的自定义品种上运行
   if(!QA_IsWhitelistedSymbol(_Symbol))
      return INIT_FAILED;

   g_board = QA_BoardOf(_Symbol);
   if(g_board == QA_BOARD_UNKNOWN)
     {
      QA_LogError(StringFormat("cannot determine the A-share board for %s", _Symbol));
      return INIT_FAILED;
     }

   //--- MQL5 正确写法：先取句柄，用时再 CopyBuffer
   if(!QA_CreateMA(g_fast, _Symbol, PERIOD_D1, InpFastPeriod, MODE_EMA, PRICE_CLOSE))
      return INIT_FAILED;
   if(!QA_CreateMA(g_slow, _Symbol, PERIOD_D1, InpSlowPeriod, MODE_EMA, PRICE_CLOSE))
      return INIT_FAILED;
   if(!QA_CreateATR(g_atr, _Symbol, PERIOD_D1, InpAtrPeriod))
      return INIT_FAILED;

   g_risk.Configure(InpMaxOrdersPerDay, InpMaxNotionalCNY,
                    InpMaxDailyLossCNY, InpMaxQuoteAgeSec);

   QA_LogInfo(StringFormat("initialised on %s board=%d tick_source=%s",
                           _Symbol, (int)g_board,
                           QA_TickSourceLabel(InpDeclaredRealTicks)));

   //--- 生成 tick 与真实 tick 必须区分报告
   if(QA_IsTesting() && !InpDeclaredRealTicks)
     {
      QA_LogWarn("Strategy Tester run WITHOUT declared real ticks: results "
                 "describe the tick generator and must not be reported as "
                 "tick-level performance.");
     }

   g_initialised = true;
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   //--- 句柄必须显式释放；漏释放会在反复加载时耗尽指标资源
   g_fast.Release();
   g_slow.Release();
   g_atr.Release();
   QA_LogInfo(StringFormat("deinit reason=%d risk=%s", reason, g_risk.Summary()));
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   if(!g_initialised || g_risk.IsKilled())
      return;

   //--- 只在有新 bar 时决策，避免在同一根 K 线上反复下单
   static datetime last_bar = 0;
   datetime current_bar = (datetime)SeriesInfoInteger(_Symbol, PERIOD_D1, SERIES_LASTBAR_DATE);
   if(current_bar == last_bar)
      return;
   last_bar = current_bar;

   //--- MQL5：句柄 -> CopyBuffer -> 数值。未就绪就退出，不用假值代替。
   double fast_value, slow_value, atr_value;
   if(!g_fast.ReadLatest(0, fast_value)) return;
   if(!g_slow.ReadLatest(0, slow_value)) return;
   if(!g_atr.ReadLatest(0, atr_value))   return;

   double fast_prev, slow_prev;
   double fast_buf[], slow_buf[];
   if(g_fast.Read(0, 1, 1, fast_buf) != 1) return;
   if(g_slow.Read(0, 1, 1, slow_buf) != 1) return;
   fast_prev = fast_buf[0];
   slow_prev = slow_buf[0];

   bool cross_up   = (fast_prev <= slow_prev) && (fast_value > slow_value);
   bool cross_down = (fast_prev >= slow_prev) && (fast_value < slow_value);

   if(!cross_up && !cross_down)
      return;

   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick))
     {
      QA_LogError("SymbolInfoTick failed");
      return;
     }
   double reference_price = (tick.last > 0.0) ? tick.last : tick.bid;
   if(reference_price <= 0.0)
     {
      QA_LogDecision(_Symbol, "SKIP", "no usable reference price");
      return;
     }

   bool have_position = PositionSelect(_Symbol);

   if(cross_up && !have_position)
      TryEnter(reference_price);
   else if(cross_down && have_position)
      TryExit(reference_price);
  }

//+------------------------------------------------------------------+
void TryEnter(const double price)
  {
   //--- A 股手数规则：股数取整，不足最小单位不下单
   double shares = QA_RoundShares(InpTargetShares, g_board, false, false);
   if(shares <= 0.0)
     {
      QA_LogDecision(_Symbol, "REJECT",
                     StringFormat("%.0f shares is below the board minimum %.0f",
                                  InpTargetShares, QA_MinimumShares(g_board)));
      return;
     }

   if(!g_risk.Allow(_Symbol, shares, price, TimeCurrent()))
      return;

   //--- 先 OrderCheck 再发送：资金/参数不合法时不要浪费一次真实请求
   MqlTradeRequest request;
   MqlTradeResult  result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action       = TRADE_ACTION_DEAL;
   request.symbol       = _Symbol;
   request.volume       = shares;      // 自定义品种 contract_size=1，volume 即股数
   request.type         = ORDER_TYPE_BUY;
   request.price        = price;
   request.deviation    = 10;
   request.type_filling = ORDER_FILLING_RETURN;
   request.comment      = "QA paper entry";

   MqlTradeCheckResult check;
   ZeroMemory(check);
   if(!OrderCheck(request, check))
     {
      QA_LogDecision(_Symbol, "REJECT",
                     StringFormat("OrderCheck failed retcode=%u %s",
                                  check.retcode, check.comment));
      return;
     }

   if(!OrderSend(request, result))
     {
      QA_LogTradeResult("entry OrderSend failed", result);
      return;
     }
   QA_LogTradeResult("entry", result);
   g_risk.Record(shares, price);
  }

//+------------------------------------------------------------------+
void TryExit(const double price)
  {
   if(!PositionSelect(_Symbol))
      return;
   double held = PositionGetDouble(POSITION_VOLUME);

   //--- 清仓允许零股 / odd lots permitted when liquidating in full
   double shares = QA_RoundShares(held, g_board, true, true);
   if(shares <= 0.0)
      return;

   if(!g_risk.Allow(_Symbol, shares, price, TimeCurrent()))
      return;

   MqlTradeRequest request;
   MqlTradeResult  result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action       = TRADE_ACTION_DEAL;
   request.symbol       = _Symbol;
   request.volume       = shares;
   request.type         = ORDER_TYPE_SELL;
   request.price        = price;
   request.deviation    = 10;
   request.type_filling = ORDER_FILLING_RETURN;
   request.comment      = "QA paper exit";

   if(!OrderSend(request, result))
     {
      QA_LogTradeResult("exit OrderSend failed", result);
      return;
     }
   QA_LogTradeResult("exit", result);
   g_risk.Record(shares, price);
  }

//+------------------------------------------------------------------+
//| 成交回报处理。不要只依赖 OrderSend 的返回值——异步成交、部分成交    |
//| 和拒单都只会在这里出现。                                            |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
  {
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
     {
      QA_LogInfo(StringFormat("deal added: symbol=%s volume=%.0f price=%.4f",
                              trans.symbol, trans.volume, trans.price));
     }
   else if(trans.type == TRADE_TRANSACTION_REQUEST)
     {
      if(result.retcode != TRADE_RETCODE_DONE &&
         result.retcode != TRADE_RETCODE_PLACED)
        {
         QA_LogWarn(StringFormat("request rejected retcode=%u %s",
                                 result.retcode, result.comment));
        }
     }
  }
//+------------------------------------------------------------------+
