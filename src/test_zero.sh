python3 main.py --solver fp --game matrixgame --generator zerosum --runner simplerunner --player "2" --actionspace "[3,3]" -t 30
python3 main.py --solver hedge --game matrixgame --generator zerosum --runner simplerunner --player "2" --actionspace "[3,3]" -t 30
python3 main.py --solver mwu --game matrixgame --generator zerosum --runner simplerunner --player "2" --actionspace "[3,3]" -t 30 --type exponential
python3 main.py --solver mwu --game matrixgame --generator zerosum --runner simplerunner --player "2" --actionspace "[3,3]" -t 30 --type linear
python3 main.py --solver rm --game matrixgame --generator zerosum --runner simplerunner --player "2" --actionspace "[3,3]" -t 30
