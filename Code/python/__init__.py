class Employee:
    empCount = 0  # 类变量，所有该类的实例共享这个变量

    # 类的构造函数，创建类的实例时，自动调用该方法
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
        Employee.empCount += 1

    def displayCount(self):
        print("Total Employee %d" % Employee.empCount)

    def displayEmployee(self):
        print("Name : ", self.name, "Salary : ", self.salary)


# 类的实例化
emp1 = Employee("Zara", 2000)
emp2 = Employee("Manni", 5000)

emp1.displayEmployee()
emp2.displayEmployee()
print(Employee.empCount)

# 添加、修改、删除类的属性
emp1.age = 18
emp1.age = 17
del emp1.age

