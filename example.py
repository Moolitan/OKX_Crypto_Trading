import okx.MarketData as MarketData
import okx.Trade as Trade

from pprint import pprint

from okx.app import OkxSWAP

import okx.Account as Account
import okx.Trade as Trade


def get_instrument_info(data, target_instId):
    for instrument in data:
        if instrument['instId'] == target_instId:
            return instrument
    return None  # 如果没有找到匹配的 instId，返回 None

# import pprint
api_key = "7e557e94-1f0f-41e1-9974-347d25487c8f"
secret_key = "D92C7D9E1B12E6D7322C36AC7DC13144"
passphrase = "Wwh199807!"

flag = "1"  # live trading: 0, demo trading: 1
# 获得市场数据
marketDataAPI = MarketData.MarketAPI(flag=flag)

result = marketDataAPI.get_tickers(instType="SWAP")


# 提取特定 instId 的信息
target_instId = 'DOGE-USDT-SWAP'
instrument_info = get_instrument_info(result["data"], target_instId)


double_data = [float(value) for key, value in instrument_info.items() if value.replace('.', '').isdigit()]

# 获得可用资金
accountAPI = Account.AccountAPI(api_key, secret_key, passphrase, False, flag)

result = accountAPI.get_account_balance()
pprint(result['data']) 

tradeAPI = Trade.TradeAPI(api_key, secret_key, passphrase, False, flag)
# 下单（市场价）
#  market order
# limit order
result = tradeAPI.place_order(
    instId="ETH-USDT",
    tdMode="cash",
    side="sell",
    ordType="market",
    sz="0.005"
)
print(result)

# if result["code"] == "0":
#     print("Successful order request，order_id = ",result["data"][0]["ordId"])
# else:
#     print("Unsuccessful order request，error_code = ",result["data"][0]["sCode"], ", Error_message = ", result["data"][0]["sMsg"])



# 使用http和https代理，proxies={'http':'xxxxx','https:':'xxxxx'}，与requests中的proxies参数规则相同
# proxies = {}
# # 转发：需搭建转发服务器，可参考：https://github.com/pyted/okx_resender
# proxy_host = None



# okxSWAP = OkxSWAP(
#     key=api_key, secret=secret_key, passphrase=passphrase, proxies=proxies, proxy_host=proxy_host,
# )

# trade = okxSWAP.trade
# # 如果有挂单或持仓，会提示“设置持仓方式为双向持仓失败”，如果你的持仓模式已经是双向持仓，可以忽略这个警告

# open_market3 = trade.open_market(
#     instId='DOGE-USDT',  # 产品
#     tdMode='cross',  # 持仓方式 isolated：逐仓 cross：全仓
#     posSide='long',  # 持仓方向 long：多单 short：空单
#     lever = 3,  # 杠杆倍数
#     openMoney = 6,  # 开仓金额 开仓金额openMoney和开仓数量quantityCT必须输入其中一个 优先级：quantityCT > openMoney
#     # quantityCT=1,  # 开仓数量 注意：quantityCT是合约的张数，不是货币数量
#     tag='',  # 订单标签
#     clOrdId='',  # 客户自定义订单ID
#     meta={},  # 向回调函数中传递的参数字典
# )

# pprint(open_market3)