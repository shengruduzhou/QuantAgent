//+------------------------------------------------------------------+
//| AShareGuards.mqh                                                  |
//| A股专用安全护栏 / A-share specific safety guards.                  |
//|                                                                   |
//| 本文件存在的理由：MT5 的默认假设来自外汇和差价合约，直接套用到      |
//| A 股现货会在三个地方出错——手数语义、T+1 结算、以及"图表上有报价    |
//| 就等于可以交易"。这里把这三件事变成显式检查。                       |
//|                                                                   |
//| Why this file exists: MT5's defaults come from FX and CFDs. Applied|
//| unchanged to A-share cash equities they are wrong about lot        |
//| semantics, T+1 settlement, and the assumption that a quoted chart  |
//| means a tradable instrument. Each becomes an explicit check here.  |
//+------------------------------------------------------------------+
#property strict

#ifndef __QUANTAGENT_ASHARE_GUARDS_MQH__
#define __QUANTAGENT_ASHARE_GUARDS_MQH__

#include <QuantAgent/Logging.mqh>

//--- 自定义品种前缀。任何非 QA_ 开头的品种都不是本仓库导入的重放数据。
//--- Custom-symbol prefix. Anything not starting QA_ is not our replay data.
#define QA_SYMBOL_PREFIX "QA_"

//--- A股板块 / A-share boards
enum ENUM_QA_BOARD
  {
   QA_BOARD_UNKNOWN = 0,
   QA_BOARD_SH_MAIN,
   QA_BOARD_SZ_MAIN,
   QA_BOARD_CHINEXT,
   QA_BOARD_STAR,
   QA_BOARD_BSE
  };

//+------------------------------------------------------------------+
//| 从 QA_600000_SH 形式的品种名推断板块。                             |
//| 只用于选择交易时段和手数规则；权威板块归属以 U0 证券主表为准。      |
//+------------------------------------------------------------------+
ENUM_QA_BOARD QA_BoardOf(const string symbol)
  {
   string body = symbol;
   if(StringFind(body, QA_SYMBOL_PREFIX) == 0)
      body = StringSubstr(body, StringLen(QA_SYMBOL_PREFIX));

   if(StringLen(body) < 6)
      return QA_BOARD_UNKNOWN;

   string code = StringSubstr(body, 0, 6);
   string head3 = StringSubstr(code, 0, 3);

   if(head3 == "688" || head3 == "689")
      return QA_BOARD_STAR;
   if(head3 == "300" || head3 == "301")
      return QA_BOARD_CHINEXT;
   if(head3 == "920" || head3 == "430" || head3 == "830" || head3 == "870")
      return QA_BOARD_BSE;
   if(StringFind(body, "_SH") >= 0)
      return QA_BOARD_SH_MAIN;
   if(StringFind(body, "_SZ") >= 0)
      return QA_BOARD_SZ_MAIN;
   return QA_BOARD_UNKNOWN;
  }

//+------------------------------------------------------------------+
//| 最小下单股数 / minimum order size in SHARES (not lots).            |
//| 科创板 200 股起、其后 1 股递增；北交所 100 股起、1 股递增；        |
//| 其余 100 股整数倍。                                                |
//+------------------------------------------------------------------+
double QA_MinimumShares(const ENUM_QA_BOARD board)
  {
   if(board == QA_BOARD_STAR)
      return 200.0;
   return 100.0;
  }

double QA_ShareStep(const ENUM_QA_BOARD board)
  {
   if(board == QA_BOARD_STAR || board == QA_BOARD_BSE)
      return 1.0;
   return 100.0;
  }

//+------------------------------------------------------------------+
//| 按板块规则取整下单股数。买入不足最小单位返回 0——不向上取整，       |
//| 因为向上取整会让回测悄悄下了一笔真实市场里下不出去的单。            |
//|                                                                   |
//| Rounds to a tradable share count. A buy below the minimum returns  |
//| 0 rather than rounding UP, because rounding up silently places an  |
//| order the real market would have rejected.                         |
//+------------------------------------------------------------------+
double QA_RoundShares(const double shares, const ENUM_QA_BOARD board,
                      const bool is_sell, const bool full_liquidation)
  {
   if(shares <= 0.0)
      return 0.0;

   double minimum = QA_MinimumShares(board);
   double step    = QA_ShareStep(board);

   if(is_sell)
     {
      //--- 零股只允许一次性清仓时卖出 / odd lots only on full liquidation
      if(full_liquidation)
         return MathFloor(shares);
      return MathFloor(shares / step) * step;
     }

   if(shares < minimum)
      return 0.0;
   return minimum + MathFloor((shares - minimum) / step) * step;
  }

//+------------------------------------------------------------------+
//| 交易时段检查 / trading-session check, Asia/Shanghai wall clock.    |
//|                                                                   |
//| 注意：MT5 返回的是"经纪商服务器时间"，不是交易所时间。调用方必须    |
//| 先完成时区换算。这里不替调用方猜时区——猜错会让整段午休变成可交易。  |
//|                                                                   |
//| NOTE: MT5 gives BROKER SERVER time, not exchange time. The caller  |
//| must convert first. This function deliberately does not guess the  |
//| offset: guessing wrong turns the entire lunch break tradable.      |
//+------------------------------------------------------------------+
bool QA_IsContinuousSession(const datetime exchange_time, const ENUM_QA_BOARD board)
  {
   MqlDateTime parts;
   TimeToStruct(exchange_time, parts);

   if(parts.day_of_week == 0 || parts.day_of_week == 6)
      return false;

   int minute_of_day = parts.hour * 60 + parts.min;

   const int am_open  =  9 * 60 + 30;   // 09:30
   const int am_close = 11 * 60 + 30;   // 11:30
   const int pm_open  = 13 * 60;        // 13:00
   const int pm_close = 14 * 60 + 57;   // 14:57 集合竞价开始

   if(minute_of_day >= am_open && minute_of_day < am_close)
      return true;
   if(minute_of_day >= pm_open && minute_of_day < pm_close)
      return true;
   return false;
  }

//+------------------------------------------------------------------+
//| 数据新鲜度检查。行情停更时策略必须停手，而不是拿着陈旧价格下单。    |
//| Stale-data guard: a strategy must stop, not trade on a frozen quote|
//+------------------------------------------------------------------+
bool QA_IsQuoteFresh(const string symbol, const int max_age_seconds)
  {
   MqlTick tick;
   if(!SymbolInfoTick(symbol, tick))
     {
      QA_LogError(StringFormat("SymbolInfoTick failed for %s", symbol));
      return false;
     }
   long age = (long)TimeCurrent() - (long)tick.time;
   if(age > max_age_seconds)
     {
      QA_LogWarn(StringFormat("%s quote is %d s old (limit %d s)",
                              symbol, (int)age, max_age_seconds));
      return false;
     }
   return true;
  }

//+------------------------------------------------------------------+
//| 实盘账户禁令 / real-account prohibition.                           |
//|                                                                   |
//| 默认拒绝真实账户。这不是建议而是硬性检查：本仓库的任何 EA 都不允许  |
//| 在真实账户上运行，除非调用方显式传入 allow_real=true，而本任务下    |
//| 没有任何 EA 会那样调用。                                            |
//|                                                                   |
//| Refuses real accounts by default. Every EA in this repository is   |
//| paper-only for this mission; none passes allow_real=true.          |
//+------------------------------------------------------------------+
bool QA_AssertNotRealAccount(const bool allow_real = false)
  {
   ENUM_ACCOUNT_TRADE_MODE mode =
      (ENUM_ACCOUNT_TRADE_MODE)AccountInfoInteger(ACCOUNT_TRADE_MODE);

   if(mode == ACCOUNT_TRADE_MODE_REAL && !allow_real)
     {
      QA_LogError("REAL account detected — QuantAgent EAs are paper-only. "
                  "Refusing to initialise.");
      return false;
     }
   if(mode == ACCOUNT_TRADE_MODE_REAL)
     {
      QA_LogError("REAL account explicitly permitted by caller. "
                  "This is outside the sanctioned configuration.");
     }
   return true;
  }

//+------------------------------------------------------------------+
//| 品种白名单 / symbol whitelist.                                     |
//| 只允许在本仓库导入的 QA_ 自定义品种上运行，避免误触经纪商真实品种。 |
//+------------------------------------------------------------------+
bool QA_IsWhitelistedSymbol(const string symbol)
  {
   if(StringFind(symbol, QA_SYMBOL_PREFIX) != 0)
     {
      QA_LogError(StringFormat(
                     "%s is not a QuantAgent custom symbol; refusing to trade "
                     "a broker instrument", symbol));
      return false;
     }
   return true;
  }

//+------------------------------------------------------------------+
//| 测试器 tick 来源判定 / real vs generated tester ticks.             |
//|                                                                   |
//| MQL5 没有直接暴露建模模式，但 TERMINAL_TRADE_ALLOWED 与            |
//| MQLInfoInteger(MQL_TESTER) 组合可以判断"是否在测试器中"。建模模式   |
//| 必须由运行方在参数里声明——本函数不猜，猜错就会把生成 tick 报告成    |
//| 真实 tick。                                                        |
//|                                                                   |
//| MQL5 does not expose the modelling mode to the EA. The runner must |
//| DECLARE it. This function refuses to infer, because inferring wrong|
//| is exactly how generated ticks get reported as real ones.          |
//+------------------------------------------------------------------+
bool QA_IsTesting()
  {
   return (bool)MQLInfoInteger(MQL_TESTER);
  }

string QA_TickSourceLabel(const bool declared_real_ticks)
  {
   if(!QA_IsTesting())
      return "LIVE_FEED";
   return declared_real_ticks ? "CUSTOM_SYMBOL_REPLAY" : "GENERATED_TESTER_TICK";
  }

#endif // __QUANTAGENT_ASHARE_GUARDS_MQH__
//+------------------------------------------------------------------+
