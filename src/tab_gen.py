from cmath import isinf
from math import isnan
import numpy as np
import pandas as pd
import os
import re

approxNEAlgos = ['kps_75', 'dmp_50',
                 'cdffjs_38', 'bbm_36', 'ts_3393', 'dfm_1_3']


approxNEAlgoNames = ['KPS06-$0.75$', 'DMP06-$0.50$',
                     'CDFFJS15-$0.38$', 'BBM07-$0.36$', 'TS07-$0.3393$', 'DFM22-$1/3$']

WSNEAlgos = ['ks_2_3', 'fgss_6605', 'cdffjs_6528', 'dfm_50']

WSNEAlgoNames = ['KS07-$2/3$',
                 'FGSS07-$0.6607$', 'CDFFJS15-$0.6528$', 'DFM22-$1/2$']

GAMUTGames = ['BertrandOligopoly', 'CournotDuopoly', 'GrabTheDollar', 'GuessTwoThirdsAve',
              'LocationGame', 'MinimumEffortGame', 'TravelersDilemma', 'WarOfAttrition']

# abbreviations for the games
GAMUTGameNames = ['BO', 'CD', 'GTD', 'GTTA', 'LG', 'MEG', 'TD', 'WOA']

ClassicGames = ['zerosum', 'random']

ClassicGameNames = ['Zero-sum', 'General']

algo_id_list = {'approxNE': approxNEAlgos, 'WSNE': WSNEAlgos}
algo_name_list = {'approxNE': approxNEAlgoNames, 'WSNE': WSNEAlgoNames}
game_id_list = {'GAMUT': GAMUTGames, 'Classic': ClassicGames}
game_name_list = {'GAMUT': GAMUTGameNames, 'Classic': ClassicGameNames}

single_row_algo_ids = ['kps_75', 'dmp_50']

no_precision_error = ['fgss_6605', 'dfm_50']

GameSizes = [10, 100, 1000]

precision = 4

long_table_thereshold = 4

# generate the body of the table
# column names: algo, game, size, last_eps, last_ws_eps, ...
df = pd.read_csv('data.csv', index_col=False)

#  print epsNE table for epsNE algorithms


def gen_table(algo_type: str, game_type: str, first_row_col_name: str, second_row_col_name: str = None, single_row_sota: bool = True) -> str:
    algo_ids = algo_id_list[algo_type]
    algo_names = algo_name_list[algo_type]
    game_ids = game_id_list[game_type]
    game_names = game_name_list[game_type]
    use_longtable = False
    # generate the header of the table
    if len(game_ids) > long_table_thereshold:
        use_longtable = True
    if use_longtable:
        tab_header = '\\begin{longtable}{c|c'
    else:
        tab_header = '''\\begin{tabular}{c|c'''
    tab_header += '|c' * len(algo_ids) + '}'
    tab_header += '''
\\toprule
\\multicolumn{2}{c|}{Scenario}'''
    for algo in algo_names:
        tab_header += ' & ' + algo
    tab_header += '''\\\\
\\midrule
'''
    if use_longtable:
        tab_header += '\\endfirsthead'
        tab_header += f'\\multicolumn{{{len(algo_ids) + 2}}}{{l}}{{\\tablename\\ \\thetable\\ -- (\\textit{{Continued}})}} \\\\\n'
        tab_header += '''\\toprule
\\multicolumn{2}{c|}{Scenario}'''
        for algo in algo_names:
            tab_header += ' & ' + algo
        tab_header += '''\\\\
\\midrule
\\endhead
'''     
        tab_header += f'\\bottomrule\n\\multicolumn{{{len(algo_ids) + 2}}}{{r}}{{\\textit{{Continued on next page}}}} \\\\\n'
        tab_header += '\\endfoot\n\\bottomrule\n\\endlastfoot\n'

    # generate the footer of the table
    tab_footer = ''
    if use_longtable:
        tab_footer += '''
\\end{longtable}
'''
    else:
        tab_footer += '''\\bottomrule
\\end{tabular}
'''
    # whether to use the textbf

    def is_sota(item: float, game: str, size: int, comp_col_name: str) -> bool:
        filtered = df[(np.isin(df['algo'], algo_ids)) &
                      (df['game'] == game) & (df['size'] == size)]
        # print(filtered)
        if single_row_sota:
            candidates = filtered[comp_col_name].to_numpy().flatten()
        else:
            candidates = filtered[[first_row_col_name,
                                   second_row_col_name]].to_numpy().flatten()
        candidates = candidates[~np.isnan(candidates)]
        # print(candidates)
        # print(np.min(candidates), item)
        return np.abs(item-np.min(candidates)) < 1e-6

    tab_body = ''
    for game in game_ids:
        tab_body += '\\multirow{6}{*}{' + \
            game_names[game_ids.index(game)] + '}'
        for size in GameSizes:
            # tab_body += ' & \\multirow{2}{*}{$' + \
            #     str(size) + '\\times ' + str(size) + '$}'
            tab_body += ' & \\multirow{2}{*}{$' + str(size) + '$}'
            for row in range(2):
                if row == 1:
                    tab_body += ' & '
                for algo in algo_ids:
                    filtered = df[(df['algo'] == algo) & (
                        df['game'] == game) & (df['size'] == size)]
                    if len(filtered) == 0:  # timeout
                        if row == 0:
                            tab_body += ' & \\multirow{2}{*}{timeout}'
                        else:
                            tab_body += ' & ~'
                    else:
                        if row == 0:  # first row
                            if algo in single_row_algo_ids:
                                tab_body += ' & \\multirow{2}{*}{'
                            else:
                                tab_body += ' & '
                            val = filtered[first_row_col_name].to_numpy().flatten()[
                                0]
                            if np.isinf(val):
                                tab_body += ' \\multirow{2}{*}{precision error}' if algo not in no_precision_error else ' \\multirow{2}{*}{timeout}'
                                continue
                            assert not isnan(
                                val), 'NaN value found in first row: ' + first_row_col_name + ', ' + algo+', ' + game + ', ' + str(size)
                            if is_sota(val, game, size, first_row_col_name):
                                tab_body += f'\\textbf{{{val:.{precision}f}}}'
                            else:
                                tab_body += f'{val:.{precision}f}'
                            if algo in single_row_algo_ids:
                                tab_body += '}'
                        else:  # second row
                            if algo in single_row_algo_ids:  # no need to print the second row
                                tab_body += ' & ~'
                            else:
                                val = filtered[second_row_col_name].to_numpy().flatten()[
                                    0]
                                if np.isinf(val):
                                    tab_body += ' & '
                                    continue
                                assert not isnan(
                                    val), 'NaN value found in second row: ' + second_row_col_name + ', ' + algo+', '+game+', '+str(size)
                                if is_sota(val, game, size, second_row_col_name):
                                    tab_body += f' & (\\textbf{{{val:.{precision}f}}})'
                                else:
                                    tab_body += f' & ({val:.{precision}f})'
                tab_body += '\\\\'
                if row == 0 or size != GameSizes[-1]:
                    tab_body += '*'
                tab_body += '\n'
    return tab_header + tab_body + tab_footer


with open('eps_of_epsNE.tex', 'w') as f:
    f.write(gen_table(algo_type='approxNE', game_type='Classic',
            first_row_col_name='last_eps', second_row_col_name='before_adjust_eps', single_row_sota=False))

with open('ws_eps_of_epsNE.tex', 'w') as f:
    f.write(gen_table(algo_type='approxNE', game_type='Classic',
            first_row_col_name='last_ws_eps', second_row_col_name='before_adjust_ws_eps', single_row_sota=False))

with open('eps_of_WSNE.tex', 'w') as f:
    f.write(gen_table(algo_type='WSNE', game_type='Classic',
            first_row_col_name='last_ws_eps', second_row_col_name='last_eps', single_row_sota=True))

with open('GAMUT_eps_of_epsNE.tex', 'w') as f:
    f.write(gen_table(algo_type='approxNE', game_type='GAMUT',
            first_row_col_name='last_eps', second_row_col_name='before_adjust_eps', single_row_sota=False))

with open('GAMUT_ws_eps_of_epsNE.tex', 'w') as f:
    f.write(gen_table(algo_type='approxNE', game_type='GAMUT',
            first_row_col_name='last_ws_eps', second_row_col_name='before_adjust_ws_eps', single_row_sota=False))

with open('GAMUT_eps_of_WSNE.tex', 'w') as f:
    f.write(gen_table(algo_type='WSNE', game_type='GAMUT',
            first_row_col_name='last_ws_eps', second_row_col_name='last_eps', single_row_sota=True))
