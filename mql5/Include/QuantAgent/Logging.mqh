//+------------------------------------------------------------------+
//| Logging.mqh                                                       |
//| 结构化日志 / structured logging with a decision trail.             |
//|                                                                   |
//| 每一次下单决策都必须留痕。测试器里 Print 会被大量调用拖慢，所以     |
//| 提供等级过滤；但 ERROR 永远输出。                                   |
//+------------------------------------------------------------------+
#property strict

#ifndef __QUANTAGENT_LOGGING_MQH__
#define __QUANTAGENT_LOGGING_MQH__

enum ENUM_QA_LOG_LEVEL
  {
   QA_LOG_ERROR = 0,
   QA_LOG_WARN  = 1,
   QA_LOG_INFO  = 2,
   QA_LOG_DEBUG = 3
  };

//--- 全局等级；EA 在 OnInit 里设置 / set by the EA in OnInit
ENUM_QA_LOG_LEVEL g_qa_log_level = QA_LOG_INFO;

void QA_SetLogLevel(const ENUM_QA_LOG_LEVEL level)
  {
   g_qa_log_level = level;
  }

void QA_LogAt(const ENUM_QA_LOG_LEVEL level, const string tag, const string message)
  {
   if(level > g_qa_log_level && level != QA_LOG_ERROR)
      return;
   PrintFormat("[QA][%s][%s] %s", tag, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS),
               message);
  }

void QA_LogError(const string message) { QA_LogAt(QA_LOG_ERROR, "ERROR", message); }
void QA_LogWarn(const string message)  { QA_LogAt(QA_LOG_WARN,  "WARN",  message); }
void QA_LogInfo(const string message)  { QA_LogAt(QA_LOG_INFO,  "INFO",  message); }
void QA_LogDebug(const string message) { QA_LogAt(QA_LOG_DEBUG, "DEBUG", message); }

//+------------------------------------------------------------------+
//| 决策留痕 / decision trail.                                         |
//| 记录"为什么下单"和"为什么不下单"，后者同样重要：一个从不解释拒单    |
//| 原因的 EA 无法审计。                                                |
//+------------------------------------------------------------------+
void QA_LogDecision(const string symbol, const string action, const string reason)
  {
   PrintFormat("[QA][DECISION] symbol=%s action=%s reason=%s", symbol, action, reason);
  }

//+------------------------------------------------------------------+
//| 交易结果留痕，含 retcode 文本 / order result with a readable code. |
//+------------------------------------------------------------------+
void QA_LogTradeResult(const string context, const MqlTradeResult &result)
  {
   PrintFormat("[QA][TRADE] %s retcode=%u deal=%I64u order=%I64u volume=%.2f "
               "price=%.4f comment=%s",
               context, result.retcode, result.deal, result.order,
               result.volume, result.price, result.comment);
  }

#endif // __QUANTAGENT_LOGGING_MQH__
//+------------------------------------------------------------------+
