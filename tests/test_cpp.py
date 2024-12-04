import ctypes  

lib = ctypes.CDLL("./paste_back.so")
 
input1 = 100
input2 = 220
result1 = lib.add(input1,input2)
result2 = lib.main()
print(result1,result2)