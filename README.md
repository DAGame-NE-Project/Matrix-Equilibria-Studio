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

### Online Learning (CCE)

1. Fictitious Play.
2. Hedge.
3. Regret Matching.
4. Multiplicative Weights Update.

### Evolutionary (ESS?)

1. Replicator Dynamics.

### Continuous Game

1. SGA.
2. Fictitious Play (Continuous Action Space).

### Deep Learning
