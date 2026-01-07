"""
高级股票监控网页应用
使用 Streamlit 和 yfinance 实现实时股票数据监控和可视化
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import json
import os
from pathlib import Path
from PIL import Image
import base64
warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="CoolDown股票监控仪表板",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 默认股票代码列表
DEFAULT_STOCKS = ["SH601727", "SH600580", "SH513010", "SZ300019", "SZ001280"]
# 基准日期
BENCHMARK_DATE = datetime(2026, 1, 1)

# 主理人数据文件路径
MANAGER_DATA_FILE = "manager_data.json"
AVATARS_DIR = "avatars"

# 确保头像目录存在
os.makedirs(AVATARS_DIR, exist_ok=True)


def convert_stock_code(code: str) -> str:
    """
    将股票代码转换为 yfinance 可识别的格式
    符合 yfinance 的国际规范：
    - 上交所：.SS (例如 601727.SS)
    - 深交所：.SZ (例如 001280.SZ)
    
    参数:
        code: 原始股票代码，如 SH601727 或 SZ300019 或 601727.SS
    
    返回:
        转换后的代码，如 601727.SS 或 300019.SZ
    """
    code = code.strip().upper()
    
    # 如果已经是正确格式（包含 .SS 或 .SZ），直接返回
    if code.endswith(".SS") or code.endswith(".SZ"):
        return code
    
    # 处理 SH 开头的上海股票代码
    if code.startswith("SH"):
        # 上海股票：去掉 SH 前缀，添加 .SS
        return code[2:] + ".SS"
    
    # 处理 SZ 开头的深圳股票代码
    elif code.startswith("SZ"):
        # 深圳股票：去掉 SZ 前缀，添加 .SZ
        return code[2:] + ".SZ"
    
    else:
        # 如果格式不明确，尝试直接返回（可能是已经正确的格式）
        return code


@st.cache_data(ttl=3600)  # 缓存1小时，避免频繁请求
def fetch_stock_data(symbol: str, start_date: datetime, end_date: datetime = None) -> pd.DataFrame:
    """
    获取股票历史数据，处理 Multi-Index 问题
    使用 yf.download 并指定 interval='1d'，同时对日期进行归一化处理
    
    参数:
        symbol: 股票代码（yfinance 格式）
        start_date: 起始日期
        end_date: 结束日期，默认为今天
    
    返回:
        处理后的 DataFrame，包含 OHLCV 数据
    """
    if end_date is None:
        end_date = datetime.now()
    
    try:
        # 归一化日期：将日期归一化到当天的 00:00:00，避免时间部分影响
        start_date_normalized = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date_normalized = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # 使用 yf.download 获取数据，指定 interval='1d'（日线数据）
        df = yf.download(
            symbol,
            start=start_date_normalized,
            end=end_date_normalized,
            interval='1d',
            progress=False
        )
        
        if df.empty:
            return pd.DataFrame()
        
        # 处理 Multi-Index：如果列是多级索引，则展平
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # 确保列名是小写，便于后续处理
        df.columns = [col.lower() for col in df.columns]
        
        # 确保索引是 DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # 统一时区处理：移除时区信息，避免后续比较时的时区问题
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # 归一化索引日期：将索引日期归一化到当天的 00:00:00
        df.index = pd.to_datetime(df.index.date)
        
        return df
    except Exception as e:
        st.error(f"获取 {symbol} 数据时出错: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_stock_info(symbol: str) -> dict:
    """
    获取股票基本信息（市值、市净率等）
    
    参数:
        symbol: 股票代码（yfinance 格式）
    
    返回:
        包含股票信息的字典
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        return {
            'market_cap': info.get('marketCap', 0),
            'pb_ratio': info.get('priceToBook', 0),
            'shares_outstanding': info.get('sharesOutstanding', 0),
            'volume_24h': info.get('volume24Hr', 0),
        }
    except Exception as e:
        return {
            'market_cap': 0,
            'pb_ratio': 0,
            'shares_outstanding': 0,
            'volume_24h': 0,
        }


def find_nearest_trading_day(df: pd.DataFrame, target_date: datetime) -> pd.Timestamp:
    """
    在数据中查找最接近目标日期的交易日
    
    参数:
        df: 股票数据 DataFrame
        target_date: 目标日期
    
    返回:
        最接近的交易日
    """
    if df.empty:
        return None
    
    # 将目标日期转换为 Timestamp
    target_ts = pd.Timestamp(target_date)
    
    # 处理时区问题：确保 target_ts 和 df.index 的时区类型一致
    if df.index.tz is not None:
        # 如果 df.index 有时区信息，将 target_ts 也转换为相同时区
        if target_ts.tz is None:
            target_ts = target_ts.tz_localize(df.index.tz)
        else:
            target_ts = target_ts.tz_convert(df.index.tz)
    else:
        # 如果 df.index 没有时区信息，确保 target_ts 也没有
        if target_ts.tz is not None:
            target_ts = target_ts.tz_localize(None)
    
    # 如果目标日期在数据范围内，尝试找到该日期或之后的第一个交易日
    if target_ts <= df.index[-1]:
        # 查找目标日期及之后的第一个交易日
        mask = df.index >= target_ts
        if mask.any():
            return df.index[mask][0]
    
    # 如果找不到之后的，找之前的最近交易日
    mask = df.index <= target_ts
    if mask.any():
        return df.index[mask][-1]
    
    # 如果都找不到，返回第一个交易日
    return df.index[0] if len(df.index) > 0 else None


def calculate_ytd_return(current_price: float, benchmark_price: float) -> float:
    """
    计算今年总体升幅（YTD Return）
    
    参数:
        current_price: 当前价格
        benchmark_price: 基准价格
    
    返回:
        涨跌幅百分比
    """
    if benchmark_price == 0 or pd.isna(benchmark_price):
        return 0.0
    return ((current_price - benchmark_price) / benchmark_price) * 100


def calculate_turnover_rate(volume: float, shares_outstanding: float) -> float:
    """
    计算换手率
    
    参数:
        volume: 成交量
        shares_outstanding: 流通股本
    
    返回:
        换手率（百分比）
    """
    if shares_outstanding == 0 or pd.isna(shares_outstanding):
        return 0.0
    return (volume / shares_outstanding) * 100


def get_all_stocks_data(stock_codes: list, benchmark_date: datetime) -> dict:
    """
    获取所有股票的数据和基准值
    
    参数:
        stock_codes: 股票代码列表
        benchmark_date: 基准日期
    
    返回:
        包含所有股票数据的字典
    """
    all_data = {}
    
    for code in stock_codes:
        yf_symbol = convert_stock_code(code)
        
        # 获取历史数据：从基准日期开始，到今天的完整数据
        # 为了确保能找到基准日，提前几天开始获取
        start_date = benchmark_date - timedelta(days=7)
        # 明确指定结束日期为今天，确保获取完整数据
        end_date = datetime.now()
        df = fetch_stock_data(yf_symbol, start_date, end_date)
        
        if df.empty:
            st.warning(f"无法获取 {code} 的数据")
            continue
        
        # 找到基准日的价格
        benchmark_trading_day = find_nearest_trading_day(df, benchmark_date)
        
        if benchmark_trading_day is None:
            st.warning(f"无法找到 {code} 的基准交易日")
            continue
        
        benchmark_price = df.loc[benchmark_trading_day, 'close']
        benchmark_market_cap = get_stock_info(yf_symbol)['market_cap']
        benchmark_pb = get_stock_info(yf_symbol)['pb_ratio']
        
        # 获取最新数据
        latest_data = df.iloc[-1]
        current_price = latest_data['close']
        current_high = latest_data['high']
        current_low = latest_data['low']
        current_volume = latest_data['volume']
        
        # 获取最新信息
        current_info = get_stock_info(yf_symbol)
        current_market_cap = current_info['market_cap']
        current_pb = current_info['pb_ratio']
        shares_outstanding = current_info['shares_outstanding']
        
        # 计算指标
        ytd_return = calculate_ytd_return(current_price, benchmark_price)
        turnover_rate = calculate_turnover_rate(current_volume, shares_outstanding)
        
        all_data[code] = {
            'yf_symbol': yf_symbol,
            'df': df,
            'benchmark_date': benchmark_trading_day,
            'benchmark_price': benchmark_price,
            'benchmark_market_cap': benchmark_market_cap,
            'benchmark_pb': benchmark_pb,
            'current_price': current_price,
            'current_high': current_high,
            'current_low': current_low,
            'current_volume': current_volume,
            'current_market_cap': current_market_cap,
            'current_pb': current_pb,
            'ytd_return': ytd_return,
            'turnover_rate': turnover_rate,
            'shares_outstanding': shares_outstanding,
        }
    
    return all_data


def create_cumulative_return_chart(all_data: dict, benchmark_date: datetime):
    """
    创建累计收益率趋势图
    
    参数:
        all_data: 所有股票数据字典
        benchmark_date: 基准日期
    
    返回:
        Plotly 图表对象
    """
    fig = go.Figure()
    
    for code, data in all_data.items():
        df = data['df']
        benchmark_price = data['benchmark_price']
        benchmark_trading_day = data['benchmark_date']
        
        # 筛选基准日之后的数据
        mask = df.index >= benchmark_trading_day
        df_filtered = df[mask].copy()
        
        if df_filtered.empty:
            continue
        
        # 计算累计收益率
        cumulative_return = ((df_filtered['close'] - benchmark_price) / benchmark_price) * 100
        
        # 添加折线
        fig.add_trace(go.Scatter(
            x=df_filtered.index,
            y=cumulative_return,
            mode='lines',
            name=code,
            line=dict(width=2),
            hovertemplate=f'<b>{code}</b><br>' +
                         '日期: %{x}<br>' +
                         '累计收益率: %{y:.2f}%<extra></extra>'
        ))
    
    # 确保x轴从基准日开始显示
    # 使用基准日作为x轴的起始点（而不是基准交易日）
    benchmark_ts = pd.Timestamp(benchmark_date)
    
    fig.update_layout(
        title='累计收益率趋势追踪图（自基准日以来）',
        xaxis_title='日期',
        yaxis_title='累计收益率 (%)',
        hovermode='x unified',
        height=500,
        # 设置x轴范围：从基准日开始，到最新数据
        xaxis=dict(
            range=[benchmark_ts, None],  # 从基准日开始，结束自动
            type='date',
            # 强制横轴不进行"毫秒级"缩放：设置日期刻度为天级别
            dtick='D1',  # 每天一个刻度
            tickformat='%Y-%m-%d',  # 日期格式：年-月-日
            tickmode='linear'  # 线性刻度模式
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white'
    )
    
    return fig


def create_comparison_chart(all_data: dict, chart_type: str, benchmark_date: datetime):
    """
    创建多维度对比图
    
    参数:
        all_data: 所有股票数据字典
        chart_type: 图表类型（'market_cap', 'volume', 'pb_ratio'）
        benchmark_date: 基准日期
    
    返回:
        Plotly 图表对象
    """
    fig = go.Figure()
    
    for code, data in all_data.items():
        df = data['df']
        benchmark_trading_day = data['benchmark_date']
        
        # 筛选基准日之后的数据
        mask = df.index >= benchmark_trading_day
        df_filtered = df[mask].copy()
        
        if df_filtered.empty:
            continue
        
        if chart_type == 'market_cap':
            # 市值变动 - 使用当前市值（因为历史市值可能无法获取）
            current_market_cap = data['current_market_cap']
            y_values = [current_market_cap] * len(df_filtered)
            y_label = '市值'
            title = '市值变动趋势'
            format_func = lambda x: f'{x/1e9:.2f}B' if x >= 1e9 else f'{x/1e6:.2f}M'
        elif chart_type == 'volume':
            # 成交量趋势
            y_values = df_filtered['volume']
            y_label = '成交量'
            title = '成交量趋势'
            format_func = lambda x: f'{x/1e6:.2f}M'
        elif chart_type == 'pb_ratio':
            # 市净率走势 - 使用当前市净率
            current_pb = data['current_pb']
            y_values = [current_pb] * len(df_filtered)
            y_label = '市净率'
            title = '市净率走势'
            format_func = lambda x: f'{x:.2f}'
        else:
            continue
        
        fig.add_trace(go.Scatter(
            x=df_filtered.index,
            y=y_values,
            mode='lines',
            name=code,
            line=dict(width=2),
            hovertemplate=f'<b>{code}</b><br>' +
                         '日期: %{x}<br>' +
                         f'{y_label}: %{{customdata}}<extra></extra>',
            customdata=[format_func(y) for y in y_values]
        ))
    
    # 确保x轴从基准日开始显示
    # 使用基准日作为x轴的起始点（而不是基准交易日）
    benchmark_ts = pd.Timestamp(benchmark_date)
    
    fig.update_layout(
        title=title,
        xaxis_title='日期',
        yaxis_title=y_label,
        hovermode='x unified',
        height=500,
        # 设置x轴范围：从基准日开始，到最新数据
        xaxis=dict(
            range=[benchmark_ts, None],  # 从基准日开始，结束自动
            type='date',
            # 强制横轴不进行"毫秒级"缩放：设置日期刻度为天级别
            dtick='D1',  # 每天一个刻度
            tickformat='%Y-%m-%d',  # 日期格式：年-月-日
            tickmode='linear'  # 线性刻度模式
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white'
    )
    
    return fig


def create_ytd_ranking_chart(all_data: dict):
    """
    创建今年总体升幅排名柱状图
    
    参数:
        all_data: 所有股票数据字典
    
    返回:
        Plotly 图表对象
    """
    codes = []
    returns = []
    
    for code, data in all_data.items():
        codes.append(code)
        returns.append(data['ytd_return'])
    
    # 按升幅排序
    sorted_data = sorted(zip(codes, returns), key=lambda x: x[1], reverse=True)
    sorted_codes, sorted_returns = zip(*sorted_data)
    
    # 设置颜色：正数红色，负数绿色
    colors = ['#FF4444' if r >= 0 else '#00AA00' for r in sorted_returns]
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(sorted_codes),
            y=list(sorted_returns),
            marker_color=colors,
            text=[f'{r:.2f}%' for r in sorted_returns],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='今年总体升幅排名',
        xaxis_title='股票代码',
        yaxis_title='累计涨跌幅 (%)',
        height=400
    )
    
    return fig


def load_manager_data() -> dict:
    """
    加载主理人数据
    
    返回:
        包含股票代码和主理人信息的字典
    """
    if os.path.exists(MANAGER_DATA_FILE):
        try:
            with open(MANAGER_DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_manager_data(data: dict):
    """
    保存主理人数据
    
    参数:
        data: 包含股票代码和主理人信息的字典
    """
    with open(MANAGER_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_manager_info(stock_code: str, manager_data: dict) -> dict:
    """
    获取股票的主理人信息
    
    参数:
        stock_code: 股票代码
        manager_data: 主理人数据字典
    
    返回:
        包含主理人姓名和头像路径的字典
    """
    if stock_code in manager_data:
        return manager_data[stock_code]
    return {'name': '', 'avatar': ''}


def save_avatar(uploaded_file, stock_code: str) -> str:
    """
    保存上传的头像文件
    
    参数:
        uploaded_file: Streamlit上传的文件对象
        stock_code: 股票代码
    
    返回:
        头像文件路径
    """
    # 获取文件扩展名
    file_ext = Path(uploaded_file.name).suffix
    # 生成文件名：股票代码_时间戳.扩展名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{stock_code}_{timestamp}{file_ext}"
    filepath = os.path.join(AVATARS_DIR, filename)
    
    # 打开并保存图片（自动处理格式转换）
    try:
        img = Image.open(uploaded_file)
        # 转换为RGB格式（如果是RGBA或其他格式）
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # 调整大小（可选，限制最大尺寸）
        img.thumbnail((200, 200), Image.Resampling.LANCZOS)
        img.save(filepath, 'JPEG', quality=95)
        return filepath
    except Exception as e:
        st.error(f"保存头像时出错: {str(e)}")
        return ""


def display_avatar(avatar_path: str, size: int = 50) -> str:
    """
    生成头像的HTML显示代码
    
    参数:
        avatar_path: 头像文件路径
        size: 显示尺寸（像素）
    
    返回:
        HTML代码字符串
    """
    if not avatar_path or not os.path.exists(avatar_path):
        return ""
    
    try:
        # 读取图片并转换为base64
        with open(avatar_path, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode()
        
        # 获取文件扩展名
        ext = Path(avatar_path).suffix.lower()
        mime_type = f"image/{ext[1:]}" if ext else "image/jpeg"
        
        return f'<img src="data:{mime_type};base64,{img_data}" width="{size}" height="{size}" style="border-radius: 50%; object-fit: cover;">'
    except:
        return ""


def format_number(num: float, is_percentage: bool = False) -> str:
    """
    格式化数字显示
    
    参数:
        num: 数字
        is_percentage: 是否为百分比
    
    返回:
        格式化后的字符串
    """
    if pd.isna(num):
        return "N/A"
    
    if is_percentage:
        return f"{num:.2f}%"
    elif abs(num) >= 1e9:
        return f"{num/1e9:.2f}B"
    elif abs(num) >= 1e6:
        return f"{num/1e6:.2f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.2f}"


# 主程序
def main():
    st.title("📈 CoolDown股票监控仪表板")
    st.markdown("---")
    
    # 加载主理人数据
    manager_data = load_manager_data()
    
    # 侧边栏：股票代码配置
    with st.sidebar:
        st.header("⚙️ 配置")
        st.subheader("股票代码设置")
        
        stock_codes = []
        for i in range(5):
            default_code = DEFAULT_STOCKS[i] if i < len(DEFAULT_STOCKS) else ""
            code = st.text_input(
                f"股票 {i+1}",
                value=default_code,
                key=f"stock_{i}"
            )
            if code:
                stock_codes.append(code)
        
        if not stock_codes:
            st.warning("请至少输入一个股票代码")
            return
        
        st.markdown("---")
        st.subheader("基准日期")
        st.info(f"基准日期：{BENCHMARK_DATE.strftime('%Y年%m月%d日')}")
        st.markdown("---")
        
        # 主理人管理
        st.subheader("👤 主理人管理")
        with st.expander("设置主理人信息", expanded=False):
            for code in stock_codes:
                st.markdown(f"**{code}**")
                manager_info = get_manager_info(code, manager_data)
                
                # 显示当前头像（如果有）
                if manager_info.get('avatar') and os.path.exists(manager_info['avatar']):
                    st.image(manager_info['avatar'], width=80, caption="当前头像")
                
                # 主理人姓名输入
                manager_name = st.text_input(
                    "主理人姓名",
                    value=manager_info.get('name', ''),
                    key=f"manager_name_{code}"
                )
                
                # 头像上传
                uploaded_avatar = st.file_uploader(
                    "上传头像",
                    type=['jpg', 'jpeg', 'png', 'gif'],
                    key=f"avatar_{code}"
                )
                
                if uploaded_avatar is not None:
                    # 保存新上传的头像
                    new_avatar_path = save_avatar(uploaded_avatar, code)
                    if new_avatar_path:
                        # 删除旧头像（如果存在且不同）
                        old_avatar = manager_info.get('avatar', '')
                        if old_avatar and old_avatar != new_avatar_path and os.path.exists(old_avatar):
                            try:
                                os.remove(old_avatar)
                            except:
                                pass
                        # 自动保存头像信息
                        manager_data[code] = {
                            'name': manager_info.get('name', ''),
                            'avatar': new_avatar_path
                        }
                        save_manager_data(manager_data)
                        st.success("头像上传成功！")
                        st.rerun()
                
                # 保存主理人姓名
                if st.button(f"保存 {code} 的主理人信息", key=f"save_{code}"):
                    manager_data[code] = {
                        'name': manager_name,
                        'avatar': manager_info.get('avatar', '')
                    }
                    save_manager_data(manager_data)
                    st.success(f"{code} 的主理人信息已保存！")
                    st.rerun()
                
                st.markdown("---")
        
        st.caption("💡 提示：数据每小时自动更新")
    
    # 获取所有股票数据
    with st.spinner("正在获取股票数据..."):
        all_data = get_all_stocks_data(stock_codes, BENCHMARK_DATE)
    
    if not all_data:
        st.error("无法获取任何股票数据，请检查股票代码是否正确")
        return
    
    # 指标卡汇总
    st.subheader("📊 指标卡汇总")
    cols = st.columns(len(all_data))
    
    for idx, (code, data) in enumerate(all_data.items()):
        with cols[idx]:
            # 获取主理人信息
            manager_info = get_manager_info(code, manager_data)
            manager_name = manager_info.get('name', '')
            avatar_path = manager_info.get('avatar', '')
            
            # 显示主理人信息（如果有）
            if manager_name or avatar_path:
                manager_cols = st.columns([1, 3])
                with manager_cols[0]:
                    if avatar_path and os.path.exists(avatar_path):
                        st.image(avatar_path, width=50, use_container_width=True)
                    else:
                        st.write("👤")
                with manager_cols[1]:
                    if manager_name:
                        st.markdown(f"**{manager_name}**")
                    else:
                        st.markdown("主理人")
            
            ytd_return = data['ytd_return']
            color = "normal" if ytd_return >= 0 else "inverse"
            st.metric(
                label=code,
                value=f"¥{data['current_price']:.2f}",
                delta=f"{ytd_return:.2f}%",
                delta_color=color
            )
    
    st.markdown("---")
    
    # 图表展示区域
    st.subheader("📈 图表分析")
    
    # 累计收益率趋势图
    st.plotly_chart(
        create_cumulative_return_chart(all_data, BENCHMARK_DATE),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # 多维度对比图
    chart_type = st.selectbox(
        "选择对比维度",
        ["市值变动", "成交量趋势", "市净率走势"],
        key="chart_type"
    )
    
    chart_type_map = {
        "市值变动": "market_cap",
        "成交量趋势": "volume",
        "市净率走势": "pb_ratio"
    }
    
    st.plotly_chart(
        create_comparison_chart(all_data, chart_type_map[chart_type], BENCHMARK_DATE),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # 今年总体升幅排名
    st.plotly_chart(
        create_ytd_ranking_chart(all_data),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # 详细数据表格
    st.subheader("📋 详细数据表格")
    
    # 准备表格数据
    table_data = []
    for code, data in all_data.items():
        # 获取主理人信息
        manager_info = get_manager_info(code, manager_data)
        manager_name = manager_info.get('name', '')
        
        table_data.append({
            '股票代码': code,
            '所属主理人': manager_name if manager_name else "未设置",
            '最新价格': f"¥{data['current_price']:.2f}",
            '最高价': f"¥{data['current_high']:.2f}",
            '最低价': f"¥{data['current_low']:.2f}",
            '市值': format_number(data['current_market_cap']),
            '市净率': f"{data['current_pb']:.2f}" if data['current_pb'] > 0 else "N/A",
            '成交量': format_number(data['current_volume']),
            '换手率': f"{data['turnover_rate']:.2f}%",
            '今年总体升幅': f"{data['ytd_return']:.2f}%",
        })
    
    df_table = pd.DataFrame(table_data)
    
    # 设置升幅列的颜色
    def color_ytd_return(val):
        try:
            return_val = float(val.replace('%', ''))
            if return_val >= 0:
                return 'background-color: #FFE5E5'  # 浅红色
            else:
                return 'background-color: #E5FFE5'  # 浅绿色
        except:
            return ''
    
    styled_df = df_table.style.applymap(
        color_ytd_return,
        subset=['今年总体升幅']
    )
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # 在主理人列下方显示头像（如果有）
    st.markdown("**主理人头像：**")
    avatar_cols = st.columns(len(all_data))
    for idx, (code, data) in enumerate(all_data.items()):
        with avatar_cols[idx]:
            manager_info = get_manager_info(code, manager_data)
            avatar_path = manager_info.get('avatar', '')
            manager_name = manager_info.get('name', '')
            
            if avatar_path and os.path.exists(avatar_path):
                st.image(avatar_path, width=80, caption=f"{code}\n{manager_name}" if manager_name else code)
            else:
                st.write(f"{code}: 无头像")
    
    # 页脚
    st.markdown("---")
    st.caption(f"最后更新：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

