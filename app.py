import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
import requests
import sqlite3
from datetime import datetime

# 1. 页面基本设置
st.set_page_config(page_title="Ultimate Quant & AI Lab", layout="wide")

# --- 全局密码登录拦截系统 ---
def check_password():
    try:
        correct_password = st.secrets["APP_PASSWORD"]
    except:
        correct_password = "123456"  # 默认密码

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"]:
        return True

    st.title("🔒 访问受限：AI 量化操盘指挥中心")
    st.markdown("为了保护您的策略与 API 资产，请输入访问密码以解锁终端。")
    
    with st.form("login_form"):
        entered_password = st.text_input("请输入访问密码", type="password")
        submit_button = st.form_submit_button("登录终端")
        
        if submit_button:
            if entered_password == correct_password:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("❌ 密码错误，请重新输入！")
                
    return False

if not check_password():
    st.stop()


# ==========================================
# 🔓 核心量化终端代码
# ==========================================

st.title("🚀 AI 量化操盘与策略实验室 (Ultimate Pro Max)")
st.markdown("多因子打分 · Alpha Vantage 异动雷达 · **策略回测引擎** · **马科维茨资产配置** · **AI 财报深度对话**")

# --- 数据库初始化 ---
def init_db():
    conn = sqlite3.connect('quant_reports.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            timestamp TEXT,
            price REAL,
            score REAL,
            report_text TEXT
        )
    ''')
    conn.commit()
    return conn

db_conn = init_db()

# 2. 侧边栏：输入与配置
st.sidebar.header("⚙️ 参数与 API 设置")
tickers = st.sidebar.text_input("输入自选资产代码 (用逗号分隔)", value="INTC, SMCI, IONQ, TSLA, GLD")
ticker_list = [t.strip().upper() for t in tickers.split(',') if t.strip()]

period = st.sidebar.selectbox("选择历史时间跨度", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

try:
    gemini_key = st.secrets["GEMINI_API_KEY"]
except:
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password")

try:
    av_key = st.secrets.get("ALPHA_VANTAGE_KEY", "")
except:
    av_key = ""

if not av_key:
    av_key = st.sidebar.text_input("Alpha Vantage API Key (可选)", type="password")


# 3. 数据批量获取与多因子量化打分函数
@st.cache_data
def fetch_all_data(tickers, period):
    data_dict = {}
    matrix_rows = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if not df.empty:
            data_dict[ticker] = df
            try:
                info = stock.info
            except:
                info = {}
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            current_price = df['Close'].iloc[-1]
            target_mean = info.get('targetMeanPrice')
            upside = ((target_mean - current_price) / current_price) * 100 if target_mean else 0
            pe = info.get("trailingPE", 0) or 0
            short_pct = info.get('shortPercentOfFloat', 0) or 0
            current_rsi = df['RSI'].iloc[-1] if not df['RSI'].empty else 50
            
            score = 50.0  
            if upside > 20: score += 20
            elif upside > 0: score += 10
            else: score -= 10
            
            if current_rsi < 30: score += 15
            elif current_rsi > 70: score -= 10
            
            if short_pct > 0.15: score -= 15
            elif short_pct < 0.05: score += 10
            
            if 0 < pe < 25: score += 15
            elif pe > 60: score -= 10
            
            score = max(0, min(100, round(score, 1)))

            matrix_rows.append({
                "资产代码": ticker,
                "当前价($)": round(current_price, 2),
                "量化综合得分": score,
                "RSI(14)": round(current_rsi, 1),
                "市盈率(PE)": round(pe, 2) if pe else "N/A",
                "预期空间": f"{upside:+.2f}%" if target_mean else "N/A",
                "做空占比": f"{short_pct*100:.2f}%"
            })
    
    matrix_df = pd.DataFrame(matrix_rows)
    if not matrix_df.empty:
        matrix_df = matrix_df.sort_values(by="量化综合得分", ascending=False)
    return data_dict, matrix_df


# 4. 双均线策略向量化回测引擎
def run_backtest(df):
    data = df.copy()
    data['MA_Short'] = data['Close'].rolling(window=20).mean()
    data['MA_Long'] = data['Close'].rolling(window=50).mean()
    data['Signal'] = np.where(data['MA_Short'] > data['MA_Long'], 1, 0)
    data['Market_Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Market_Returns'] * data['Signal'].shift(1)
    
    data['Cum_Market'] = (1 + data['Market_Returns'].fillna(0)).cumprod() - 1
    data['Cum_Strategy'] = (1 + data['Strategy_Returns'].fillna(0)).cumprod() - 1
    
    total_strat_return = data['Cum_Strategy'].iloc[-1] * 100
    total_mkt_return = data['Cum_Market'].iloc[-1] * 100
    
    rolling_max = (1 + data['Strategy_Returns'].fillna(0)).cumprod().cummax()
    drawdown = (1 + data['Strategy_Returns'].fillna(0)).cumprod() / rolling_max - 1
    max_dd = drawdown.min() * 100
    
    return data, total_strat_return, total_mkt_return, max_dd


# 5. 马科维茨资产配置优化器 (MPT)
def optimize_portfolio(data_dict):
    prices = pd.DataFrame({ticker: df['Close'] for ticker, df in data_dict.items()}).dropna()
    if prices.shape[1] < 2:
        return None, "资产数量少于2个，无法计算组合配置。"
    
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    num_assets = len(prices.columns)
    np.random.seed(42)
    
    best_sharpe = -np.inf
    best_weights = None
    
    for _ in range(5000):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = weights
            
    res_df = pd.DataFrame({
        "资产": prices.columns,
        "最优配置权重(%)": [round(w * 100, 2) for w in best_weights]
    })
    return res_df, f"模拟最优夏普比率 (Sharpe): {round(best_sharpe, 2)}"


# 6. 获取 Alpha Vantage 新闻与数据库函数
@st.cache_data(ttl=3600)
def fetch_av_news(api_key, ticker):
    if not api_key: return [], "未配置 Key"
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}&limit=3"
    try:
        res = requests.get(url, timeout=4)
        data = res.json()
        if "feed" in data: return data["feed"], "Alpha Vantage 实时 API"
    except: pass
    return [], "降级方案"

def save_report_to_db(ticker, price, score, text):
    cursor = db_conn.cursor()
    cursor.execute("INSERT INTO reports (ticker, timestamp, price, score, report_text) VALUES (?, ?, ?, ?, ?)",
                   (ticker, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), price, score, text))
    db_conn.commit()

def load_reports_from_db(ticker):
    cursor = db_conn.cursor()
    cursor.execute("SELECT timestamp, price, score, report_text FROM reports WHERE ticker = ? ORDER BY id DESC", (ticker,))
    return cursor.fetchall()


# 7. AI 深度分析引擎
def run_ai_analysis(api_key, ticker, info, current_price, target_mean, news_items, score):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3.5-flash')
        news_summary = "".join([f"- [{i.get('publisher','媒体')}] {i.get('title','无标题')}\n" for i in news_items[:4]])
        prompt = f"""
        你是一位顶尖华尔街量化总监。请基于以下数据对【{ticker}】进行操盘决断：
        - 量化综合得分: {score}/100 | 当前价: ${current_price:.2f} | 目标均价: ${target_mean if target_mean else 'N/A'}
        - PE: {info.get('trailingPE', 'N/A')} | 做空比例: {f"{info.get('shortPercentOfFloat', 0)*100:.2f}%" if info.get('shortPercentOfFloat') else 'N/A'}
        - 最新新闻: {news_summary if news_summary else '暂无'}
        请输出：1.量化得分点评 2.操盘手最终裁决(买入/观望/卖出) 3.风险边界。
        """
        return model.generate_content(prompt).text
    except Exception as e:
        return f"❌ 生成失败: {str(e)}"


# 8. 主流程渲染
if ticker_list:
    with st.spinner("正在加载量化数据与实验室模型..."):
        stock_data_dict, matrix_df = fetch_all_data(ticker_list, period)

    # 顶层布局：左右分栏
    main_col, control_col = st.columns([7, 3])

    with control_col:
        st.markdown("### 🎛️ 操盘指挥与智能晨报")
        if av_key:
            _, status_msg = fetch_av_news(av_key, ticker_list[0])
            st.success(f"🟢 状态: {status_msg}")
        else:
            st.info("⚪ 状态: 未配置 Alpha Vantage Key")

        st.divider()
        if st.button("📧 生成今日操盘早报"):
            if gemini_key:
                genai.configure(api_key=gemini_key)
                brief = genai.GenerativeModel('gemini-3.5-flash').generate_content(f"为自选股 {ticker_list} 写一份200字精炼早报").text
                st.info(brief)

        st.markdown("### 🏆 AI 投资推荐雷达")
        
        # 完整的 5 大宏观赛道选项
        rec_sector = st.selectbox(
            "选择宏观赛道", 
            [
                "🚀 AI 算力与半导体产业链", 
                "⚡ 新能源汽车与具身智能机器人", 
                "💰 全球大宗商品、黄金与避险配置", 
                "🧬 生物医药与创新科技突破",
                "🌐 全球宏观高股息与防御性资产"
            ],
            key="rec_selector"
        )

        if st.button("🔥 生成该赛道 AI 投资推荐列表"):
            if not gemini_key:
                st.warning("请先配置 Gemini API Key")
            else:
                with st.spinner(f"AI 正在深度扫描【{rec_sector}】的市场动向..."):
                    genai.configure(api_key=gemini_key)
                    model = genai.GenerativeModel('gemini-3.5-flash')
                    rec_res = model.generate_content(f"请为{rec_sector}赛道推荐 3 只顶级资产（包含股票代码与名称），并详细说明买入理由、操盘评级与潜在风险。").text
                    st.success("✅ 推荐列表已生成")
                    st.info(rec_res)

    with main_col:
        # 多因子天梯榜
        if not matrix_df.empty:
            st.subheader("🔥 多因子量化综合评分天梯榜 (Quant Score)")
            st.dataframe(matrix_df, use_container_width=True, hide_index=True)
            
            if len(ticker_list) >= 2:
                opt_df, opt_msg = optimize_portfolio(stock_data_dict)
                if opt_df is not None:
                    with st.expander("⚖️ 马科维茨多资产最优配置方案 (Portfolio Optimization)"):
                        st.caption(opt_msg)
                        st.dataframe(opt_df, use_container_width=True, hide_index=True)
            st.divider()

        if stock_data_dict:
            for ticker, df in stock_data_dict.items():
                st.header(f"📊 {ticker} 深度量化与策略回测台")
                stock = yf.Ticker(ticker)
                
                try:
                    info = stock.info
                except:
                    info = {}
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("当前市值", f"${info.get('marketCap', 0) / 1e9:.2f}B" if info.get('marketCap') else "N/A")
                c2.metric("市盈率 PE", round(info.get("trailingPE", 0), 2) if info.get("trailingPE") else "N/A")
                c3.metric("利润率", f"{info.get('profitMargins', 0) * 100:.2f}%" if info.get('profitMargins') else "N/A")
                c4.metric("机构评级", info.get('recommendationKey', 'N/A').upper())

                # --- K线与策略回测图表 ---
                df['50_MA'] = df['Close'].rolling(window=50).mean()
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K线"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['50_MA'], line=dict(color='orange', width=2), name='50MA'), row=1, col=1)
                
                target_mean = info.get('targetMeanPrice')
                if target_mean:
                    fig.add_hline(y=target_mean, line_dash="dash", line_color="green", annotation_text=f"目标价:${target_mean}", row=1, col=1)

                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=1.5), name='RSI'), row=2, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color="blue", row=2, col=1)
                fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(t=20, b=20, l=10, r=10))
                st.plotly_chart(fig, use_container_width=True)

                # --- 策略回测看板 ---
                with st.expander(f"📈 运行 {ticker} 双均线策略回测 (Backtest Engine)"):
                    bt_df, strat_ret, mkt_ret, max_dd = run_backtest(df)
                    bc1, bc2, bc3 = st.columns(3)
                    bc1.metric("双均线策略收益率", f"{strat_ret:+.2f}%", delta=f"对比大盘 {mkt_ret:+.2f}%")
                    bc2.metric("最大回撤 (Max DD)", f"{max_dd:.2f}%")
                    bc3.metric("回测状态", "正常运行")
                    
                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Cum_Strategy'], name="双均线策略净值", line=dict(color='blue')))
                    fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Cum_Market'], name="买入持有(大盘)净值", line=dict(color='gray', dash='dash')))
                    fig_bt.update_layout(title="策略累计收益净值曲线对比", height=300, margin=dict(t=30, b=10, l=10, r=10))
                    st.plotly_chart(fig_bt, use_container_width=True)

                current_price = df['Close'].iloc[-1]
                current_score = matrix_df.loc[matrix_df["资产代码"] == ticker, "量化综合得分"].values[0] if not matrix_df.empty else 50

                # --- AI 深度分析与 SQLite 数据库 ---
                st.markdown("### 🤖 Gemini AI 操盘决策与智能财报问答")
                
                if not gemini_key:
                    st.warning("请配置 Gemini API Key")
                else:
                    if st.button(f"🚀 运行 {ticker} AI 操盘推演并归档", key=f"btn_{ticker}"):
                        with st.spinner("AI 正在推演并写入本地 SQLite..."):
                            report = run_ai_analysis(gemini_key, ticker, info, current_price, target_mean, stock.news, current_score)
                            save_report_to_db(ticker, current_price, current_score, report)
                            st.success("✅ 研报已成功写入本地 SQLite！")
                            st.info(report)

                # --- AI 财报专属对话框 ---
                user_query = st.chat_input(f"💬 向 Gemini 追问关于 {ticker} 的财报、估值或风险细节...", key=f"chat_{ticker}")
                if user_query and gemini_key:
                    genai.configure(api_key=gemini_key)
                    chat_model = genai.GenerativeModel('gemini-3.5-flash')
                    context_prompt = f"你是资产{ticker}的专属华尔街研究员。当前价格${current_price}，PE {info.get('trailingPE')}。用户问：{user_query}"
                    reply = chat_model.generate_content(context_prompt).text
                    st.markdown(f"**🤖 AI 助手回答：** {reply}")

                # 本地数据库历史回溯
                with st.expander(f"📂 查看 {ticker} 的历史 AI 操盘研报存档 (SQLite)"):
                    history_rows = load_reports_from_db(ticker)
                    if history_rows:
                        for h_time, h_price, h_score, h_text in history_rows:
                            st.markdown(f"**🕒 存档：{h_time} | 价格：${h_price} | 得分：{h_score}分**")
                            st.markdown(h_text)
                            st.divider()
                    else:
                        st.caption("暂无历史存档。")

                st.divider()
        else:
            st.error("未能获取资产数据。")

update app.py with full features
