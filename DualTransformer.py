import axelrod as axl
from axelrod.strategy_transformers import dual

p2 = axl.TitForTat()
a = axl.WinStayLoseShift()
p1 = dual(a)
m = axl.Tournament([p1, p2], turns=5, repetitions=10)
m.play(processes=2)
print("Tournament Complete")
p3 = axl.TitForTat()
p4 = dual(axl.WinStayLoseShift())
o = axl.Match([p4, p3], 5)
o.play()
print(o.result)

