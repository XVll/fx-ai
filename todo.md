- How do I handle non closed trades, now we have 2048 steps per episode, which we start at random point in a day. Then
  what will we do open positions for example model has + 2000$ pnl but havent closed it yet then episode truncates ?

- Model keeps buying at the ask and selling on the bid non-stop how do we prevent that (we do not added spread features
  yet)

- Sweep has errors when starting, fix them.

Right now our configuration system bothering me, feeling like scattered and cannot find where is where also, I dont even know if there is configs not used or used anymore.
Also I am not sure if any of the configs used but not in config files. Also in the project some uses yaml some uses typed configs so it become a mess.
previously I tried to add typed configurations but now. I have to write both yaml and typed configs, do you have suggestion how should I design better config system for the
project givme suggestions so I can choose one. I like to have warnings if config is exists when used or, do I using configs defined in yaml files.   