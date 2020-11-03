# SPoon
**SPoon** is a shortest path calc tool using some algorithms with cuda to speedup.

It's **developing**.



## 开发日志

### 10/29

- 完成了前期工作的总结和整合。
- 并新建了仓库。
- 测试了异步协同编程



### 10/30

- 绘制了新流程图
- 依据新流程修改程序流程与接口



### 10/31

- 对修改后的包进行简单测试
- 重新修改parameter中的属性
- 对接了读入函数
- debug delta-stepping & edge



### 11/1

- 集成化测试了各个用例
- 编写了使用算法的伪代码
- 增加了block和grid的属性参数



### 11/2

- 完成了自动化测试用例

- 修复了函数接口错误

- 完善分图路由，并测试和修复了分图中bug

  

### 11/3

- 完成了新结构代码的注释编写
- 在每个函数中添加了logger功能
- 去除了调整结构后的一些bug
- 添加了requirement
- 完成了path recording功能
- 重构和设计了Result类
- 去除了无用的另一些文件和方法
- 实现了文本display函数，展示本次计算的相关信息
- 实现了绘路径图功能 但是图还不理想

- [ ] 但SPFA多源出现了奇怪的bug，每次只计算了一个源
- [ ] 绘制的图还得漂亮



## ToDo

- [ ] fw 或者 matrix 给跑起来
- [ ] 开发者文档
- [ ] 用法截图
- [ ] 大规模集成测试
- [ ] 简单usage