import numpy as np

class MW(object):

    def __init__(self):
        pass

    def _multiplicative_weights(game, mix, func, epsilon):
        """Generic multiplicative weights algorithm
        Parameters
        ----------
        game : RsGame
            The game to compute an equilibrium of.
        mix : ndarray
            The initial mixture for searching.
        func : (RsGame, ndarray) -> ndarray
            Function that takes a game and a mixture and returns an unbiased
            estimate of the payoff to each strategy when opponents play according
            to the mixture.
        epsilon : float
            The rate of update for new payoffs. Convergence results hold when
            epsilon in [0, 3/5].
        """
        average = mix.copy()
        # This is done in log space to prevent weights zeroing out for limit cycles
        with np.errstate(divide="ignore"):
            log_weights = np.log(mix)
        learning = np.log(1 + epsilon)

        for i in itertools.count(2):  # pragma: no branch
            pays = func(game, np.exp(log_weights))
            log_weights += pays * learning
            log_weights -= np.logaddexp.reduceat(log_weights, game.role_starts).repeat(
                game.num_role_strats
            )
            average += (np.exp(log_weights) - average) / i
            yield average


    def _mw_dist(game, mix):
        """Distributional multiplicative weights payoff function"""
        return game.deviation_payoffs(mix)


    def multiplicative_weights_dist(
        game, mix, *, epsilon=0.5, max_iters=10000, converge_thresh=1e-8, **kwargs
    ):
        """Compute an equilibrium using the distribution multiplicative weights
        This version of multiplicative weights takes the longest per iteration, but
        also has less variance and likely converges better.
        Parameters
        ----------
        game : RsGame
            The game to compute an equilibrium of.
        mix : ndarray
            The initial mixture for searching.
        epsilon : float, optional
            The rate of update for new payoffs. Convergence results hold when
            epsilon in [0, 3/5].
        """
        return _multiplicative_weights(  # pylint: disable=unexpected-keyword-arg
            game,
            mix,
            _mw_dist,
            epsilon,
            max_iters=max_iters,
            converge_thresh=converge_thresh,
            converge_disc=1,
            **kwargs
        )
