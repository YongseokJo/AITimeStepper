## Progress



## Todo
- apply it to the three body sim
- different particle masses

## Strategy
- Evergy and/or momentum and/or angular momentum conservation (Unsupervised)
- Trajectory based (Supervised) -- I dont think this will be the option anymore. This is only applies to this particular two body example, which is not that generalizable at all.
- I can inject past positions and acceleration
- I can predict future acceleration (extra info) and use it to estimate to timestep.
- one thing note is that when it comes to circular orbit, the direction of accelerations changes but the magnitude of acceleration does not change. In this case, there can be a lot of degeneracies.
- I can feed surrounding particles within certain radius as well.


## Notes
- the same energy conservation can have different orbits.
- I can do mass loss/gain case such as accreting black holes or star losing mass by winds.
- Another test is "under tidal field".