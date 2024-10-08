# Matrix-Equilibria-Studio

This project is to build a Nash equilibrium toolkit for matrix games.
For matrix games, we provide several game generators for different game types, including Prison Dilemma, Chicken Game, Rock-Paper-Scissors, etc.
For equilibrium solvers, we divide them into two different kinds, one-shot solver and online solver.

## Matrix Game

### Attributes

`generator` a generator to generate the game.

`players` the number of players.

`U[a]` the utility (or utility distribution) function when the joint action of agents is `a` and `U[a, id]` is the utility (or utility distribution) function of agent `id`.

`actionspace` is a tuple of the action spaces for agents, and `actionspace[id]` is the action space of agent `id`.

### Functions

`constructor(generator = default)` generates a game with the generator (maybe with params).

`play(a)` or `playID(a, ID)` returns the utility `U[a]` or `U[a, ID]`, respectively.

`getActionSpace` or `getActionSpaceID(id)` is a function to get the action space.

`reset(generator = None)` is a function to regenerate the game, when the parameter `generator` is not `None`, `self.generator` will be replaced by the new one (maybe with params).

## Solver

### Functions

`solve(game, info)` given a game and some info, return a strategy and new info.

## Algorithm

### Traditional (NE)

1. Support Enumeration. 
2. Lemke-Howson.

### Approximate NE

1. KPS-3/4 
2. DMP-0.5 
3. CDFFJS-0.38 
4. DMP-0.38 
5. BBM-0.36 
6. TS-0.3393 
7. DFM-1/3

### Special Case NE

1. solve zero-sum games by LP

### Online Learning (CCE)

1. Fictitious Play.
2. Hedge.
3. Regret Matching.
4. Multiplicative Weights Update.

### Well-supported Nash Equilibrium (WSNE)

1. KS-2/3
2. FGSS-0.6607
3. CDFFJS-0.6528
4. DFM-1/2

### Evolutionary (ESS)

1. Replicator Dynamics.
