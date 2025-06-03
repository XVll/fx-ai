- How do I handle non closed trades, now we have 2048 steps per episode, which we start at random point in a day. Then
  what will we do open positions for example model has + 2000$ pnl but havent closed it yet then episode truncates ?

- Model keeps buying at the ask and selling on the bid non-stop how do we prevent that (we do not added spread features
  yet)

- Sweep has errors when starting, fix them.


Okay do not implement anything yet but I want you to analyze our new curriculum learning, I refactored whole training cycle and moved all logic that decides what to train on and how long to there to make it easier for hyper parameter       │
│   tuning. First I want you to analyze it and create a plan for that implementation, if that is not apporiate for this task just say how we can fix it. In current version we are able to select symbol, date range and end conditions like        │
│   episode, update, cycle. we can also choose which stages or single stage with unlimited training etc, we can choose the day quality, reset point quality etc. Chack if it is compatible with our next hyper parameter optimization and create a  │
│   plan.  