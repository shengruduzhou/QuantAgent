//+------------------------------------------------------------------+
//| RiskGuard.mqh                                                     |
//| 交易前风控与熔断 / pre-trade risk limits and kill switch.          |
//|                                                                   |
//| 设计原则：风控失败必须**阻止**下单，而不是记一条日志然后继续。      |
//| 每个限额都有明确数值和当前测量值，拒单理由可审计。                  |
//|                                                                   |
//| Design rule: a failed check BLOCKS the order rather than logging   |
//| and continuing. Every limit carries both its threshold and the     |
//| measured value, so a rejection is auditable.                       |
//+------------------------------------------------------------------+
#property strict

#ifndef __QUANTAGENT_RISK_GUARD_MQH__
#define __QUANTAGENT_RISK_GUARD_MQH__

#include <QuantAgent/Logging.mqh>
#include <QuantAgent/AShareGuards.mqh>

//+------------------------------------------------------------------+
//| 风控状态。EA 在 OnInit 里配置限额，每次下单前调用 Allow()。         |
//+------------------------------------------------------------------+
class QARiskGuard
  {
private:
   int      m_max_orders_per_day;
   double   m_max_notional_cny;
   double   m_max_daily_loss_cny;
   int      m_max_quote_age_seconds;

   int      m_orders_today;
   double   m_notional_today;
   double   m_day_start_equity;
   datetime m_current_day;
   bool     m_killed;
   string   m_kill_reason;

   datetime DayOf(const datetime when) const
     {
      MqlDateTime parts;
      TimeToStruct(when, parts);
      parts.hour = 0; parts.min = 0; parts.sec = 0;
      return StructToTime(parts);
     }

public:
                     QARiskGuard(void)
      : m_max_orders_per_day(20), m_max_notional_cny(1000000.0),
        m_max_daily_loss_cny(20000.0), m_max_quote_age_seconds(120),
        m_orders_today(0), m_notional_today(0.0), m_day_start_equity(0.0),
        m_current_day(0), m_killed(false), m_kill_reason("") {}

   void Configure(const int max_orders, const double max_notional,
                  const double max_daily_loss, const int max_quote_age)
     {
      m_max_orders_per_day    = max_orders;
      m_max_notional_cny      = max_notional;
      m_max_daily_loss_cny    = max_daily_loss;
      m_max_quote_age_seconds = max_quote_age;
     }

   //--- 跨日重置。没有这一步，昨天的计数会永久占用今天的额度。
   void RollDay(const datetime now)
     {
      datetime today = DayOf(now);
      if(today != m_current_day)
        {
         m_current_day      = today;
         m_orders_today     = 0;
         m_notional_today   = 0.0;
         m_day_start_equity = AccountInfoDouble(ACCOUNT_EQUITY);
         QA_LogInfo(StringFormat("risk guard rolled to a new day, start equity %.2f",
                                 m_day_start_equity));
        }
     }

   bool IsKilled(void) const { return m_killed; }
   string KillReason(void) const { return m_kill_reason; }

   void Kill(const string reason)
     {
      m_killed      = true;
      m_kill_reason = reason;
      QA_LogError(StringFormat("KILL SWITCH ENGAGED: %s", reason));
     }

   //--- 当日亏损熔断 / daily-loss circuit breaker
   void CheckDailyLoss(void)
     {
      if(m_day_start_equity <= 0.0)
         return;
      double equity = AccountInfoDouble(ACCOUNT_EQUITY);
      double loss   = m_day_start_equity - equity;
      if(loss > m_max_daily_loss_cny)
         Kill(StringFormat("daily loss %.2f exceeds limit %.2f",
                           loss, m_max_daily_loss_cny));
     }

   //+---------------------------------------------------------------+
   //| 下单前总检查。返回 false 即禁止下单，并已写明原因。              |
   //+---------------------------------------------------------------+
   bool Allow(const string symbol, const double shares, const double price,
              const datetime exchange_time)
     {
      RollDay(TimeCurrent());
      CheckDailyLoss();

      if(m_killed)
        {
         QA_LogDecision(symbol, "REJECT", "kill switch: " + m_kill_reason);
         return false;
        }
      if(!QA_IsWhitelistedSymbol(symbol))
        {
         QA_LogDecision(symbol, "REJECT", "symbol not on the QA_ whitelist");
         return false;
        }
      if(!QA_IsQuoteFresh(symbol, m_max_quote_age_seconds))
        {
         QA_LogDecision(symbol, "REJECT", "stale quote");
         return false;
        }
      if(!QA_IsContinuousSession(exchange_time, QA_BoardOf(symbol)))
        {
         QA_LogDecision(symbol, "REJECT", "outside the continuous session");
         return false;
        }
      if(m_orders_today >= m_max_orders_per_day)
        {
         QA_LogDecision(symbol, "REJECT",
                        StringFormat("order count %d reached the daily limit %d",
                                     m_orders_today, m_max_orders_per_day));
         return false;
        }

      double notional = shares * price;
      if(m_notional_today + notional > m_max_notional_cny)
        {
         QA_LogDecision(symbol, "REJECT",
                        StringFormat("notional %.2f would exceed the daily cap %.2f "
                                     "(used %.2f)", notional, m_max_notional_cny,
                                     m_notional_today));
         return false;
        }

      return true;
     }

   //--- 下单成功后登记用量 / register consumption after a successful send
   void Record(const double shares, const double price)
     {
      m_orders_today++;
      m_notional_today += shares * price;
     }

   string Summary(void) const
     {
      return StringFormat("orders=%d/%d notional=%.2f/%.2f killed=%s",
                          m_orders_today, m_max_orders_per_day,
                          m_notional_today, m_max_notional_cny,
                          m_killed ? "yes" : "no");
     }
  };

#endif // __QUANTAGENT_RISK_GUARD_MQH__
//+------------------------------------------------------------------+
