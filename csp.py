from constraint import *

p = Problem()
p.addVariable("a", [2])
p.addVariable("b", [2])


# p.addVariable("c",[1,2,3])

print("without constrains", p.getSolutions())
p.addConstraint(lambda a, b: a != b, ["a", "b"])
print()

print("with constrain", p.getSolutions())



