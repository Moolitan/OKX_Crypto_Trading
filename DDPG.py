import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import time
import okx.Account as Account
from datetime import datetime
from typing import Dict, List, Union
import okx.MarketData as MarketData
import okx.Trade as Trade
# import pprint
api_key = "7e557e94-1f0f-41e1-9974-347d25487c8f"
secret_key = "D92C7D9E1B12E6D7322C36AC7DC13144"
passphrase = "Wwh199807!"
flag = "1"  # live trading: 0, demo trading: 1
target_instId = 'ETH-USDT-SWAP'
# ==================== 数据预处理模块 ====================
class DataPreprocessor:
    def __init__(self):
        self.feature_ranges = [
            (0, 1e5),    # last
            (0, 1e4),    # lastSz
            (0, 1e5),    # askPx
            (0, 1e4),    # askSz
            (0, 1e5),    # bidPx
            (0, 1e4),    # bidSz
            (0, 1e5),    # open24h
            (0, 1e5),    # high24h
            (0, 1e5),    # low24h
            (0, 1e9),    # volCcy24h
            (0, 1e9),    # vol24h
            (0, 1e18),   # ts
            (0, 1e5),    # sodUtc0
            (0, 1e5)     # sodUtc8
        ]
    
    def preprocess(self, raw_data):
        processed = []
        for i, value in enumerate(raw_data):
            min_val, max_val = self.feature_ranges[i]
            processed.append((value - min_val) / (max_val - min_val))
        
        # 添加时间特征
        ts = raw_data[11] / 1000
        processed.append((ts % 86400) / 86400)
        return np.array(processed, dtype=np.float32)

# ==================== DDPG网络结构 ====================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # 输出范围[-1, 1]
        )
    
    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        self.action_net = nn.Linear(action_dim, 256)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, action):
        state_out = self.state_net(state)
        action_out = self.action_net(action)
        return self.fc(torch.cat([state_out, action_out], dim=1))

# ==================== 交易环境与风险管理 ====================
class TradingEnvironment:

    def __init__(self):
        # 账户核心状态
        self.cash: float = 0.0             # USDT可用余额
        self.position: float = 0.0         # 标的资产持仓（如ETH数量）
        self.total_equity: float = 0.0     # 总资产估值(USD)
        self.unrealized_pnl: float = 0.0   # 未实现盈亏
        self.last_updated: str = ""        # 最后更新时间
        self.peak_equity = 10000.0
        self.initial_balance = 10000.0  # 在环境初始化时设置
        self.max_position = 10000.0       # 根据标的资产设置最大预期持仓

        self.equity_history = []
        self.position_history = []
        self.trade_count = 0 

        # 币种详细信息
        self.currencies: List[Dict] = []   # 各币种信息列表


        

        # 初始化时立即同步账户数据
        self.update_from_api_response([])  # 传入空列表初始化

    def update_from_api_response(self, api_response: List[Dict]):
        """
        从OKX API响应更新账户状态
        :param api_response: API返回的原始账户数据
        """
        try:
            accountAPI = Account.AccountAPI(api_key, secret_key, passphrase, False, flag)
            result = accountAPI.get_account_balance()
            # 解析顶层字段
            top_level = result['data'] if result else {}
            self._parse_top_level(top_level[0])
            
            # 解析币种详细信息
            self.currencies = self._parse_currency_details(top_level[0].get("details", []))
            
            # 更新核心交易状态
            self._update_trading_state()
            
        except (IndexError, KeyError, TypeError) as e:
            print(f"[Warning] 账户更新失败，保持最后有效状态. 错误: {str(e)}")

    def _parse_top_level(self, data: Dict):
        """解析顶层字段"""
        # 总资产估值
        self.total_equity = self._safe_float(data.get("totalEq"))
        
        # 未实现盈亏
        self.unrealized_pnl = self._safe_float(data.get("upl"))
        
        # 更新时间处理
        raw_ts = self._safe_int(data.get("uTime"))
        self.last_updated = self._timestamp_to_str(raw_ts)

    def _parse_currency_details(self, details: List[Dict]) -> List[Dict]:
        """解析币种详细信息"""
        parsed = []
        for curr in details:
            item = {
                "ccy": curr.get("ccy", "UNKNOWN"),
                "available": self._safe_float(curr.get("availBal")),
                "equity_usd": self._safe_float(curr.get("eqUsd"))
            }
            parsed.append(item)
        
        # 按权益价值降序排序
        return sorted(parsed, key=lambda x: x["equity_usd"], reverse=True)

    def _update_trading_state(self):
        """更新交易相关状态"""
        # 提取USDT现金
        usdt = next(
            (c for c in self.currencies if c["ccy"] == "USDT"),
            {"available": 0.0, "equity_usd": 0.0}
        )
        self.cash = usdt["available"]
        
        # 提取标的资产持仓（示例为ETH）
        eth = next(
            (c for c in self.currencies if c["ccy"] == "ETH"),
            {"available": 0.0}
        )
        self.position = eth["equity_usd"]

    @staticmethod
    def _safe_float(value: Union[str, float, None]) -> float:
        """安全转换为浮点数"""
        try:
            return float(value) if value not in [None, ""] else 0.0
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _safe_int(value: Union[str, int, None]) -> int:
        """安全转换为整数"""
        try:
            return int(value) if value not in [None, ""] else 0
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def _timestamp_to_str(ts: int) -> str:
        """时间戳转换 (兼容毫秒/秒)"""
        try:
            ts = ts // 1000 if ts > 1e12 else ts
            return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S UTC')
        except:
            return "1970-01-01 00:00:00 UTC"

    def get_account_summary(self) -> str:
        """生成账户摘要信息"""
        return (
            f"账户状态 ({self.last_updated}):\n"
            f"• 总资产: ${self.total_equity:,.2f}\n"
            f"• 未实现盈亏: ${self.unrealized_pnl:+,.2f}\n"
            f"• 可用现金(USDT): {self.cash:.2f}\n"
            f"• 标的持仓: {self.position:.6f}\n"
            f"• 资产分布:\n" + 
            "\n".join([
                f"  {c['ccy']}: {c['available']:.8f} (${c['equity_usd']:.2f})" 
                for c in self.currencies if c['equity_usd'] > 0
            ])
        )


    

   
          
    def execute_trade(self, action,current_price):
        """
        action: 一维动作值（范围[-1,1]）
        - -1~0 : 卖出持仓比例（绝对值）
        - 0    : 不操作
        - 0~1  : 买入可用资金比例（限制单次最大1000U且不超过2%）
        """
        current_price = current_price  # 当前价格
        
        # 卖出逻辑（-1 ≤ action < 0）
        if -1 <= action < 0:
            if self.position > 0.4:  # 有持仓才能卖
                # 计算卖出比例（取绝对值）
                sell_ratio = abs(action)
                sell_amount = sell_ratio * self.position / current_price
                
                # # 执行卖出
                # self.cash += sell_amount * current_price
                # self.position -= sell_amount
                # print(f"[卖出] 比例 {sell_ratio:.2%} | 数量 {sell_amount:.4f} | 获得 ${sell_amount*current_price:.2f}")

                tradeAPI = Trade.TradeAPI(api_key, secret_key, passphrase, False, flag)
                result = tradeAPI.place_order(
                    instId="ETH-USDT",
                    tdMode="cash",
                    side="sell",
                    ordType="market",
                    sz=str(sell_amount.item())
                )
                self.trade_count += 1
            else:
                print("[警告] 尝试卖出但无持仓")
                

        # 买入逻辑（0 < action ≤ 1）
        elif 0 < action <= 1:
            if self.cash > 0:  # 有现金才能买
                # 计算可用资金上限（2%和1000U取小）
                max_cash = min(np.array([self.cash * 0.02]), np.array([2]))
                
                # 实际买入金额 = 动作比例 * 可用上限
                invest_cash = action * max_cash

                
                # 执行买入（确保不超出现金余额）
                buy_amount = max(min(invest_cash, self.cash),np.array([0.4]))

                # buy_amount = actual_cost / current_price
                
                # self.cash -= actual_cost
                # self.position += buy_amount
                # print(f"[买入] 比例 {action:.2%} | 花费 ${actual_cost:.2f} | 获得 {buy_amount:.4f} 单位")
                # tradeAPI = Trade.TradeAPI(api_key, secret_key, passphrase, False, flag)
                # 下单（市场价）
                #  market order
                #  limit order

                tradeAPI = Trade.TradeAPI(api_key, secret_key, passphrase, False, flag)
                result = tradeAPI.place_order(
                    instId="ETH-USDT",
                    tdMode="cash",
                    side="buy",
                    ordType="market",
                    sz=str(buy_amount.item())
                )
                self.trade_count += 1
            else:
                print("[警告] 尝试买入但现金不足")

        # action=0 不操作
        else:
            self.trade_count = 0
            pass  # 无操作
        # 更新净值历史



    
    def calculate_risk(self):
        # 计算最大回撤
        equity = np.array(self.equity_history[-100:] or [0])
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # 计算波动率
        returns = np.diff(np.log(equity + 1e-9))
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        
        return max_drawdown, volatility

# ==================== DDPG训练框架 ====================
class DDPGTrainer:
    def __init__(self):
        # 环境参数
        self.state_dim = 15 + 5  # 市场特征 + 账户特征
        self.action_dim = 1      # [买入比例, 卖出比例]
        
        # 初始化组件
        self.actor = Actor(self.state_dim, self.action_dim)
        self.actor_target = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim, self.action_dim)
        self.critic_target = Critic(self.state_dim, self.action_dim)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.replay_buffer = deque(maxlen=50000)
        self.preprocessor = DataPreprocessor()
        self.env = TradingEnvironment()
        
        # 训练参数
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_sizes = [150, 500, 2000]
        self.risk_coef = 0.2  # 风险惩罚系数
        self.profit_coef = 1.0         # 收益系数（可调参数）


    

    def _get_state(self, raw_data):

        self.env.update_from_api_response([])

        equity = self.env.total_equity
        self.env.equity_history.append(equity)
        self.env.position_history.append(self.env.position)
        self.env.peak_equity = max(self.env.peak_equity, equity)

        processed = self.preprocessor.preprocess(raw_data)

        account_state = [
            self.env.cash / max(self.env.initial_balance, 1e-8),  # 现金比例（防止除以零）
            self.env.position / max(self.env.max_position, 1e-8), # 仓位比例
            (self.env.peak_equity - self.env.initial_balance) / max(self.env.initial_balance, 1),  # 净值回撤率
            self.env.total_equity / max(self.env.initial_balance, 1),  # 总资产比例
            self.env.unrealized_pnl / max(self.env.total_equity, 1)    # 盈亏占比（基于当前总资产）
        ]
        return account_state, np.concatenate([processed, np.clip(account_state, -5, 5)])
    
    def _add_exploration_noise(self, action):
        noise = torch.normal(0, 0.1, size=action.shape)
        return torch.clamp(action + noise, -1, 1)
    
    def train_step(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        
        # 更新Critic
        with torch.no_grad():
            target_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, target_actions)
            target_q = rewards + self.gamma * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        # 更新Actor
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        # 软更新目标网络
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def get_instrument_info(self, data, target_instId):
        for instrument in data:
            if instrument['instId'] == target_instId:
                return instrument
        return None  # 如果没有找到匹配的 instId，返回 None
    
    def run_training(self, total_episodes=1000000):
        for episode in range(1, total_episodes+1):
            # 获得市场数据
            marketDataAPI = MarketData.MarketAPI(flag=flag)
            result = marketDataAPI.get_tickers(instType="SWAP")
            # 提取特定 instId 的信息
            
            instrument_info = self.get_instrument_info(result["data"], target_instId)
            double_data = [float(value) for key, value in instrument_info.items() if value.replace('.', '').isdigit()]

            account_information,state = self._get_state(double_data)
            
            # 生成动作
            with torch.no_grad():
                action = self.actor(torch.FloatTensor(state).unsqueeze(0))[0].detach().cpu().numpy()  # 先分离梯度再转换
                # action = self._add_exploration_noise(torch.FloatTensor(action)).numpy()

            # 执行交易
            # action = -0.99999
            self.env.execute_trade(action, double_data[0])

            # 获取新状态
            # instrument_info = self.get_instrument_info(result["data"], target_instId)

            double_data = [float(value) for key, value in instrument_info.items() if value.replace('.', '').isdigit()]

            next_account_information,next_state = self._get_state(double_data)
            
            # 计算带风险惩罚的奖励
            current_equity = next_account_information[0]
            profit = current_equity - account_information[0]  # 绝对收益

        
            trade_penalty = 0.01 * self.env.trade_count  # 每笔交易惩罚0.01%
            max_drawdown, volatility = self.env.calculate_risk()
            if volatility > 0.2:  # 高波动市场
                self.risk_coef = 1.0
                self.profit_coef = 0.5
            else:                 # 低波动市场
                self.risk_coef = 0.5
                self.profit_coef = 1.0

             # 组合奖励函数
            reward = 100 * (
                self.profit_coef * profit                # 收益激励
                - self.risk_coef * (max_drawdown * 10000 + volatility * 100)  # 风险惩罚
                - trade_penalty                          # 交易频率惩罚
            )

            reward = np.clip(reward, -1e6, 1e6)
            
            # 存储经验
            self.replay_buffer.append((state, action, reward, next_state))
            
            # 分级训练
            if len(self.replay_buffer) > 200:
                self.train_step(128)
            if len(self.replay_buffer) > 1000 and episode % 100 == 0:
                self.train_step(256)
            if len(self.replay_buffer) > 10000 and episode % 500 == 0:
                self.train_step(1024)
            
            # 保存模型
            if episode % 1000 == 0:
                torch.save({
                    'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict()
                }, f"ddpg_model_ep{episode}.pth")

            print("Episode: ", episode, 
                  ' Reward : ', reward , 
                  " Before_equity : ",account_information[0],
                  " Total_equity : ",  next_account_information[0], 
                  " Action: ", action)
            time.sleep(3)  # API频率限制
            

# ==================== 执行训练 ====================
if __name__ == "__main__":
    trainer = DDPGTrainer()
    trainer.run_training()