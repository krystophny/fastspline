# Manual calculation of fprati for p1=1, f1=2, p2=2, f2=1, p3=3, f3=-1

p1, f1, p2, f2, p3, f3 = 1.0, 2.0, 2.0, 1.0, 3.0, -1.0

print('Manual fprati calculation:')
print(f'p1={p1}, f1={f1}, p2={p2}, f2={f2}, p3={p3}, f3={f3}')

# Since p3 = 3 > 0, use case p3 != infinity (lines 16-19)
h1 = f1 * (f2 - f3)  # 2 * (1 - (-1)) = 2 * 2 = 4
h2 = f2 * (f3 - f1)  # 1 * (-1 - 2) = 1 * (-3) = -3  
h3 = f3 * (f1 - f2)  # -1 * (2 - 1) = -1 * 1 = -1

print(f'h1 = f1*(f2-f3) = {f1}*({f2}-({f3})) = {h1}')
print(f'h2 = f2*(f3-f1) = {f2}*({f3}-{f1}) = {h2}')
print(f'h3 = f3*(f1-f2) = {f3}*({f1}-{f2}) = {h3}')

# p = -(p1*p2*h3 + p2*p3*h1 + p3*p1*h2) / (p1*h1 + p2*h2 + p3*h3)
numerator = -(p1*p2*h3 + p2*p3*h1 + p3*p1*h2)
denominator = p1*h1 + p2*h2 + p3*h3

print(f'numerator = -(p1*p2*h3 + p2*p3*h1 + p3*p1*h2)')
print(f'          = -({p1}*{p2}*{h3} + {p2}*{p3}*{h1} + {p3}*{p1}*{h2})')
print(f'          = -({p1*p2*h3} + {p2*p3*h1} + {p3*p1*h2})')
print(f'          = -{p1*p2*h3 + p2*p3*h1 + p3*p1*h2} = {numerator}')

print(f'denominator = p1*h1 + p2*h2 + p3*h3')
print(f'            = {p1}*{h1} + {p2}*{h2} + {p3}*{h3}')
print(f'            = {p1*h1} + {p2*h2} + {p3*h3}')  
print(f'            = {denominator}')

if denominator != 0:
    p = numerator / denominator
    print(f'p = {numerator} / {denominator} = {p}')
else:
    print('Division by zero!')

# Final adjustment based on f2 sign (line 21-26)
print(f'f2 = {f2}')
if f2 >= 0:  # f2 >= 0 case (go to line 22-24)
    print('f2 >= 0, so p1=p2, f1=f2')
    p1_out = p2
    f1_out = f2
    p3_out = p3
    f3_out = f3
else:  # f2 < 0 case (go to line 25-26)
    print('f2 < 0, so p3=p2, f3=f2')
    p1_out = p1
    f1_out = f1
    p3_out = p2  
    f3_out = f2

print(f'Final: p={p}, p1_out={p1_out}, f1_out={f1_out}, p3_out={p3_out}, f3_out={f3_out}')