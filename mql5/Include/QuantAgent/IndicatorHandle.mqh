//+------------------------------------------------------------------+
//| IndicatorHandle.mqh                                               |
//| 指标句柄生命周期 / indicator handle lifecycle.                     |
//|                                                                   |
//| 这是 MQL4 代码迁到 MQL5 最常出错的地方。MQL4 里 iMA(...) 直接返回   |
//| 数值，MQL5 里返回的是句柄，必须再用 CopyBuffer 取值，并且句柄要在    |
//| OnDeinit 里 IndicatorRelease。把 MQL4 写法照抄过来会得到一个恒等于  |
//| 句柄编号的"指标值"——通常是个很小的整数，看起来像价格，非常难发现。  |
//|                                                                   |
//| This is where ported MQL4 code most often breaks. In MQL4 iMA()    |
//| returns a VALUE; in MQL5 it returns a HANDLE that must be read via |
//| CopyBuffer and released in OnDeinit. Copying MQL4 syntax across    |
//| yields an "indicator value" that is really a small integer handle  |
//| id — which looks plausibly like a price and is very hard to spot.  |
//+------------------------------------------------------------------+
#property strict

#ifndef __QUANTAGENT_INDICATOR_HANDLE_MQH__
#define __QUANTAGENT_INDICATOR_HANDLE_MQH__

#include <QuantAgent/Logging.mqh>

//+------------------------------------------------------------------+
//| RAII-ish wrapper：构造取句柄，析构释放。                            |
//+------------------------------------------------------------------+
class QAIndicator
  {
private:
   int      m_handle;
   string   m_name;

public:
                     QAIndicator(void) : m_handle(INVALID_HANDLE), m_name("") {}
                    ~QAIndicator(void) { Release(); }

   //--- 绑定一个已创建的句柄 / adopt a handle created by the caller
   bool Adopt(const int handle, const string name)
     {
      Release();
      m_handle = handle;
      m_name   = name;
      if(m_handle == INVALID_HANDLE)
        {
         QA_LogError(StringFormat("indicator %s: handle creation failed, error %d",
                                  name, GetLastError()));
         return false;
        }
      return true;
     }

   bool IsValid(void) const { return m_handle != INVALID_HANDLE; }
   int  Handle(void)  const { return m_handle; }
   string Name(void)  const { return m_name; }

   void Release(void)
     {
      if(m_handle != INVALID_HANDLE)
        {
         IndicatorRelease(m_handle);
         m_handle = INVALID_HANDLE;
        }
     }

   //+---------------------------------------------------------------+
   //| 读取缓冲区。返回实际拷贝的条数；不足请求量时**不**静默补零，     |
   //| 因为补零会让"指标还没算好"看起来像"指标值等于 0"。               |
   //|                                                                |
   //| Returns the count actually copied. A short read is NOT padded   |
   //| with zeros: padding makes "indicator not ready yet" look        |
   //| identical to "indicator value is zero".                         |
   //+---------------------------------------------------------------+
   int Read(const int buffer_index, const int start, const int count,
            double &out[]) const
     {
      if(!IsValid())
        {
         QA_LogError(StringFormat("indicator %s: read on an invalid handle", m_name));
         return -1;
        }
      ArraySetAsSeries(out, true);
      int copied = CopyBuffer(m_handle, buffer_index, start, count, out);
      if(copied < 0)
        {
         QA_LogError(StringFormat("indicator %s: CopyBuffer failed, error %d",
                                  m_name, GetLastError()));
         return -1;
        }
      if(copied < count)
        {
         QA_LogDebug(StringFormat("indicator %s: only %d of %d bars available",
                                  m_name, copied, count));
        }
      return copied;
     }

   //--- 便捷单值读取；未就绪返回 false，绝不返回一个假值。
   bool ReadLatest(const int buffer_index, double &value) const
     {
      double buffer[];
      if(Read(buffer_index, 0, 1, buffer) != 1)
         return false;
      value = buffer[0];
      return true;
     }
  };

//+------------------------------------------------------------------+
//| 工厂函数 / factories. 每个都返回句柄，由 QAIndicator::Adopt 接管。  |
//+------------------------------------------------------------------+
bool QA_CreateMA(QAIndicator &out, const string symbol, const ENUM_TIMEFRAMES tf,
                 const int period, const ENUM_MA_METHOD method,
                 const ENUM_APPLIED_PRICE price)
  {
   int handle = iMA(symbol, tf, period, 0, method, price);
   return out.Adopt(handle, StringFormat("MA(%d)", period));
  }

bool QA_CreateATR(QAIndicator &out, const string symbol, const ENUM_TIMEFRAMES tf,
                  const int period)
  {
   int handle = iATR(symbol, tf, period);
   return out.Adopt(handle, StringFormat("ATR(%d)", period));
  }

bool QA_CreateRSI(QAIndicator &out, const string symbol, const ENUM_TIMEFRAMES tf,
                  const int period, const ENUM_APPLIED_PRICE price)
  {
   int handle = iRSI(symbol, tf, period, price);
   return out.Adopt(handle, StringFormat("RSI(%d)", period));
  }

bool QA_CreateMACD(QAIndicator &out, const string symbol, const ENUM_TIMEFRAMES tf,
                   const int fast, const int slow, const int signal,
                   const ENUM_APPLIED_PRICE price)
  {
   int handle = iMACD(symbol, tf, fast, slow, signal, price);
   return out.Adopt(handle, StringFormat("MACD(%d,%d,%d)", fast, slow, signal));
  }

bool QA_CreateBands(QAIndicator &out, const string symbol, const ENUM_TIMEFRAMES tf,
                    const int period, const double deviation,
                    const ENUM_APPLIED_PRICE price)
  {
   int handle = iBands(symbol, tf, period, 0, deviation, price);
   return out.Adopt(handle, StringFormat("Bands(%d,%.1f)", period, deviation));
  }

//+------------------------------------------------------------------+
//| 自定义指标 / custom indicator via iCustom.                         |
//+------------------------------------------------------------------+
bool QA_CreateCustom(QAIndicator &out, const string symbol, const ENUM_TIMEFRAMES tf,
                     const string indicator_path, const int param)
  {
   int handle = iCustom(symbol, tf, indicator_path, param);
   return out.Adopt(handle, StringFormat("iCustom(%s)", indicator_path));
  }

#endif // __QUANTAGENT_INDICATOR_HANDLE_MQH__
//+------------------------------------------------------------------+
