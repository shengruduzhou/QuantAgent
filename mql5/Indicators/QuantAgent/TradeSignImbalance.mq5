//+------------------------------------------------------------------+
//| TradeSignImbalance.mq5                                            |
//| 成交方向失衡 / trade-sign imbalance over a rolling window.         |
//|                                                                   |
//| 定义 / definition:                                                 |
//|     imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
//|                                                                   |
//| 关于输入数据的诚实说明 / an honest note on the input:              |
//|                                                                   |
//| 本指标的方向来自 tick flags 里的 TICK_FLAG_BUY / TICK_FLAG_SELL。   |
//| 对于本仓库导入的 A 股自定义品种，这些 flag 源自公开数据源自己的     |
//| 方向分类（腾讯的 B/S/M 标记），**不是交易所发布的主动买卖方向**。    |
//| 也就是说这是一个 quote-rule 推断值，不是观测值。真正的成交方向需要  |
//| Level-2 逐笔成交里的买卖委托编号，而那需要券商 QMT 权限。            |
//|                                                                   |
//| The direction here comes from tick flags, which for our imported   |
//| A-share symbols originate in the public vendor's own B/S/M         |
//| classification, NOT an exchange-published aggressor side. It is an |
//| inference, not an observation. A genuine trade side needs the bid/ |
//| ask order numbers in Level-2 逐笔成交, which requires a broker QMT |
//| entitlement this repository does not hold.                        |
//|                                                                   |
//| 因此：本指标可用于研究相对变化，不可用于宣称"订单流"结论。          |
//+------------------------------------------------------------------+
#property copyright "QuantAgent"
#property version   "1.00"
#property strict

#property indicator_separate_window
#property indicator_buffers 2
#property indicator_plots   2

#property indicator_label1  "Imbalance"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrDodgerBlue
#property indicator_width1  2

#property indicator_label2  "Zero"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrGray
#property indicator_style2  STYLE_DOT

#property indicator_minimum -1.0
#property indicator_maximum  1.0

input int InpWindowBars = 20;   // 滚动窗口（K线数）

double ImbalanceBuffer[];
double ZeroBuffer[];

//+------------------------------------------------------------------+
int OnInit()
  {
   SetIndexBuffer(0, ImbalanceBuffer, INDICATOR_DATA);
   SetIndexBuffer(1, ZeroBuffer,      INDICATOR_DATA);
   IndicatorSetString(INDICATOR_SHORTNAME,
                      StringFormat("QA TradeSignImbalance(%d) [inferred side]",
                                   InpWindowBars));
   IndicatorSetInteger(INDICATOR_DIGITS, 4);

   if(InpWindowBars < 2)
     {
      Print("[QA][ERROR] InpWindowBars must be >= 2");
      return INIT_PARAMETERS_INCORRECT;
     }
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
   if(rates_total < InpWindowBars)
      return 0;

   int start = (prev_calculated > 0) ? prev_calculated - 1 : InpWindowBars;

   for(int i = start; i < rates_total; i++)
     {
      ZeroBuffer[i] = 0.0;

      //--- 用收盘方向近似成交方向：本 bar 收涨记为买、收跌记为卖。
      //--- 这是 bar 级别的粗近似，比直接假装拥有逐笔方向要诚实。
      double buy_volume  = 0.0;
      double sell_volume = 0.0;

      for(int k = 0; k < InpWindowBars; k++)
        {
         int index = i - k;
         if(index <= 0)
            break;
         //--- real_volume 优先：自定义品种把股数写在这里
         double bar_volume = (volume[index] > 0) ? (double)volume[index]
                                                 : (double)tick_volume[index];
         if(close[index] > close[index - 1])
            buy_volume += bar_volume;
         else if(close[index] < close[index - 1])
            sell_volume += bar_volume;
         //--- 平盘不计入任何一边，而不是随意归给某一方
        }

      double total = buy_volume + sell_volume;
      ImbalanceBuffer[i] = (total > 0.0) ? (buy_volume - sell_volume) / total : 0.0;
     }

   return rates_total;
  }
//+------------------------------------------------------------------+
