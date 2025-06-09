# DAY1实习课程笔记
**（下列内容为本日实习课程笔记）**
## 1. 环境准备
```bash
# 检查 Python 版本
python --version

# 创建并激活虚拟环境
python -m venv myenv

# Windows
myenv\Scripts\activate

# Linux / macOS
source myenv/bin/activate

# 安装第三方库
pip install requests
```

---

## 2. 变量、变量类型与作用域
- **基本类型**：`int`, `float`, `str`, `bool`, `list`, `tuple`, `dict`, `set`
- **作用域**：全局变量、局部变量，`global` 和 `nonlocal`
- **类型转换**：`int()`, `str()`, `float()`, `list()` 等
```python
name = "Alice"  # str
age = 20        # int
grades = [90, 85, 88]  # list
info = {"name": "Alice", "age": 20}  # dict

# 类型转换
age_str = str(age)
number = int("123")

# 作用域示例
x = 10
def my_function():
    global x
    x += 1
    y = 5
    print(f"Inside: x={x}, y={y}")

my_function()
print(f"Outside: x={x}")
```

---

## 3. 运算符与表达式
- **算术运算符**：`+`, `-`, `*`, `/`, `//`, `%`, `**`
- **比较运算符**：`==`, `!=`, `>`, `<`, `>=`, `<=`
- **逻辑运算符**：`and`, `or`, `not`
- **位运算符**：`&`, `|`, `^`, `<<`, `>>`
```python
a = 10
b = 3
print(a + b, a // b, a ** b)

x = True
y = False
print(x and y, x or y)

print(a > b)
```

---

## 4. 条件、循环与异常处理
- **条件语句**：`if`, `elif`, `else`
- **循环语句**：`for`, `while`, `break`, `continue`
- **异常处理**：`try`, `except`, `finally`
```python
score = 85
if score >= 90:
    print("A")
elif score >= 60:
    print("Pass")
else:
    print("Fail")

for i in range(5):
    if i == 3:
        continue
    print(i)

try:
    num = int(input("Enter a number: "))
    print(100 / num)
except ZeroDivisionError:
    print("Cannot divide by zero!")
except ValueError:
    print("Invalid input!")
finally:
    print("Done.")
```

---

## 5. 函数与高阶函数
- **定义函数**：`def`
- **参数类型**：默认参数、可变参数（`*args`, `**kwargs`）
- **匿名函数**：`lambda`
- **高阶函数**：函数作为参数或返回值
```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

print(greet("Alice"))
print(greet("Bob", "Hi"))

def sum_numbers(*args):
    return sum(args)

double = lambda x: x * 2

def apply_func(func, value):
    return func(value)

print(apply_func(lambda x: x ** 2, 4))
```

---

## 6. 模块与包
- **模块导入**：`import`, `from ... import ...`
- **自定义模块**：`.py` 文件
- **包结构**：含 `__init__.py` 的文件夹
- **常用第三方库**：`requests`, `numpy`, `pandas`
```python
# mymodule.py
def say_hello():
    return "Hello from module!"

# 使用模块
import mymodule
print(mymodule.say_hello())

# 第三方模块
import requests
response = requests.get("https://api.github.com")
print(response.status_code)
```

---

## 7. 类与对象
- **类定义**：`class`
- **面向对象三大特性**：封装、继承、多态
```python
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        return f"I am {self.name}, {self.age} years old."

class GradStudent(Student):
    def __init__(self, name, age, major):
        super().__init__(name, age)
        self.major = major

    def introduce(self):
        return f"I am {self.name}, a {self.major} student."

student = Student("Alice", 20)
grad = GradStudent("Bob", 22, "CS")
print(student.introduce())
print(grad.introduce())
```

---

## 8. 装饰器
- **本质**：函数嵌套 + 高阶函数
- **语法糖**：`@decorator`
- **带参数的装饰器**
```python
def my_decorator(func):
    def wrapper():
        print("Before")
        func()
        print("After")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()

def repeat(n):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hi, {name}!")

greet("Alice")
```

---

## 9. 文件操作
- **文本文件**：`open()`, `read()`, `write()`
- **上下文管理器**：`with`
- **CSV 与 JSON 文件处理**
```python
# 写入文件
with open("example.txt", "w") as f:
    f.write("Hello, Python!\n")

# 读取文件
with open("example.txt", "r") as f:
    content = f.read()
    print(content)

# CSV 文件
import csv
with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Age"])
    writer.writerow(["Alice", 20])
```

---

## 10. Git基础命令
```bash
# 初始化仓库
git init

# 添加文件到暂存区
git add .

# 提交更改
git commit -m "提交信息"

# 添加远程仓库
git remote add origin "远程地址"

# 拉取并变基
git pull --rebase origin main

# 推送到远程仓库
git push origin main

# 配置用户名和邮箱
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"

# 配置代理
git config --global http.proxy http://127.0.0.1:53950
```

---

## 11.Conda 安装与库管理
```bash
# 安装 conda 后创建环境
conda create -n myenv python=3.10

# 激活虚拟环境
conda activate myenv

# 安装库
conda install numpy pandas matplotlib

# 检查Conda版本
conda

# 更新Conda本身
conda update conda

# 显示所有环境
conda env list

# 删除指定环境
conda env remove -n myenv

# 导出环境配置文件
conda env export > environment.yml

# 根据配置文件创建环境
conda env create -f environment.yml
```

---
