"""
This will handle order placement/fills, commissions, fees, etc.
Commissions and Fees: The environment will deduct a commission cost per trade (and possibly market impact fees or short borrow fees for short positions). These costs are small per trade but significant for high frequency and should be included in reward calculationsgithub.com.
"""


While implementing imagine that we have generic, data layer that process both live,historic data and apply pre process for features. You can use that as imaginery provider. It will provide. 1s, 1m, 5m data, trades(tape), quote(top of book) and LuLD ofc I will process even further like adding indicators for like to 1m and 5min, analyzing trades and quote etc. Also imageine we have engines that simulates market environment and order execution like market_simulator, order_execution. You can use them too with the functions/features you need you also do not need to implement these but use as an interface. While implementing I want you to do it with great detail and also provide comments that expains what does what. Especially the parameters, configurations that needs my attention so I can configure for my needs.