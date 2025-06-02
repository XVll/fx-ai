- How do I handle non closed trades, now we have 2048 steps per episode, which we start at random point in a day. Then
  what will we do open positions for example model has + 2000$ pnl but havent closed it yet then episode truncates ?

- Model keeps buying at the ask and selling on the bid non-stop how do we prevent that (we do not added spread features
  yet)

- Sweep has errors when starting, fix them.


 thinkultra, think. We are having issues with reward system and actions. Model finds a way to abuse what every I do, It first started not taking trades because of losses, I added inactivity penalty , then it started not taking trades and 
  spamming sell action which is not possible. I want you to consider action masking, and a new reward design so it cannot abuse and will allow to train model in best possible way. Now 
  create a new structure for rewards and masking based on our model design and trading style. Such design, will also allow to tune for begginer to advanced. With parameters over time we will make it harder but first need to learn trade. 
  Reward selections must be perfect and smart. Now write down your plan without implementing.