# a=[2,4,6,1,10,4]
# my_dict = dict(name="lowman", age=45, money=998, hourse=None)
# print(my_dict)
# key_list = my_dict.values()
# print(list(key_list))


# 列表如下

a=[
{'name': 'eth0', 'macAddr': 2},
 {'name': 'eth5', 'macAddr': 74},
 {'name': 'eth3', 'macAddr':15 },
 {'name': 'eth1', 'macAddr': 7},
{'name': 'eth4', 'macAddr': 6},
{'name': 'eth2', 'macAddr': 76}
]
# 取出每个列表key为macAddr的对应value，组成列表，例如这种
# ["2c:44:fd:7f:56:a4","d8:9d:67:f2:ef:74".......]
#
# 代码：
lis = [ i['macAddr']  for i in a ]
lis_a=max(lis)
b=lis.index(lis_a)
print(lis_a)
print(b)
m=list(a[5].values())
print(m)
n=m[0]
sd=m[1]
print(n)