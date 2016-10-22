import axelrod as axl
from axelrod.strategy_transformers import DualTransformer

p2 = axl.TitForTat()
p1 = DualTransformer(axl.WinStayLoseShift())(axl.WinStayLoseShift)()
m = axl.Match([p1, p2], 5)
m.play()
print(m.result)
