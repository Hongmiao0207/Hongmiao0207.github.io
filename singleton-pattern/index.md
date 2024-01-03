# Singleton Pattern for Java


## 创建型模式

- 用于描述"怎么创建对象"，特点是**将对象的创建和使用分离**。
- 如：单例、原型、工厂方法、抽象工厂、建造者等。

## Singleton Pattern

涉及到一个单一的类，负责创建自己的的对象，同时保证只有单个对象被创建，并提供一个访问其唯一对象的方式来访问，而不需要被额外实例化。

### 单例模式的结构

- 单例类：只创建一个实例的类。
- 访问类：使用单例类。

### 1.1.2 单例模式的实现

分类两种：

- 饿汉式
- 懒汉式

#### 饿汉式

1. 饿汉方式1：静态变量方式

```java
public class Singleton {
    //私有构造方法
    private Singleton() {}
​
    //在成员位置创建该类的对象
    private static Singleton instance = new Singleton();
​
    //对外提供静态方法获取该对象
    public static Singleton getInstance() {
        return instance;
    }
}
```

在成员位置声明 Singleton类型的静态变量，并创建Singleton类的对象instance。instance对象是随着类的加载而创建的。

弊端：如果长期不使用就会造成内存浪费。

2. 饿汉方式2：静态代码块

```java
public class Singleton {
​
    //私有构造方法
    private Singleton() {}
​
    //在成员位置创建该类的对象
    private static Singleton instance;
​
    static {
        instance = new Singleton();
    }
​
    //对外提供静态方法获取该对象
    public static Singleton getInstance() {
        return instance;
    }
}
```

3. 饿汉方式3：枚举

```java
public enum Singleton {
    INSTANCE;
}
```

枚举类型时线程安全的，并且只会装载一次。另外，其写法非常简单，是所有单例实现中唯一不会被破坏的单例实现模式。

在成员位置声明Singleton类型的静态变量，而对象的创建是在静态代码块中，同样也是随着类的加载而创建。因此弊端同方式1一样。

#### 懒汉式

1. 懒汉方式1：线程不安全

```java
public class Singleton {
    //私有构造方法
    private Singleton() {}
​
    //在成员位置创建该类的对象
    private static Singleton instance;
​
    //对外提供静态方法获取该对象
    public static Singleton getInstance() {
​
        if(instance == null) {
            instance = new Singleton();
        }
        return instance;
    }
}
```

在成员位置声明Singletonde类型的静态变量，在调用getInstance()时，Singleton类的对象为null时才创建，实现了懒加载效果。在多线程环境下，有线程安全问题。

2. 懒汉方式2：线程安全

```java
public class Singleton {
    //私有构造方法
    private Singleton() {}
​
    //在成员位置创建该类的对象
    private static Singleton instance;
​
    //对外提供静态方法获取该对象
    public static synchronized Singleton getInstance() {
​
        if(instance == null) {
            instance = new Singleton();
        }
        return instance;
    }
}
```

在getInstance()加了一个同步关键字synchonized，虽然解决安全问题，但是执行效率低（为了第一次的初始化线程安全问题而牺牲性能）。

3. 懒汉方式3：双重检查锁

```java
public class Singleton { 
​
    //私有构造方法
    private Singleton() {}
​
    private static Singleton instance;
​
   //对外提供静态方法获取该对象
    public static Singleton getInstance() {
        //第一次判断，如果instance不为null，不进入抢锁阶段，直接返回实例
        if(instance == null) {
            synchronized (Singleton.class) {
                //抢到锁之后再次判断是否为null
                if(instance == null) {
                    instance = new Singleton();
                }
            }
        }
        return instance;
    }
}
```

解决了性能问题，又保证了线程安全。但在多线程下，可能会出现空指针问题（因为JVM在实例化对象时会进行优化和指令重排操作）。

可以通过volatile关键字来解决双重锁带来的空指针异常问题，volatile可以保证可见性和有序性。

```java
public class Singleton {
​
    //私有构造方法
    private Singleton() {}
​
    private static volatile Singleton instance;
​
   //对外提供静态方法获取该对象
    public static Singleton getInstance() {
        //第一次判断，如果instance不为null，不进入抢锁阶段，直接返回实际
        if(instance == null) {
            synchronized (Singleton.class) {
                //抢到锁之后再次判断是否为空
                if(instance == null) {
                    instance = new Singleton();
                }
            }
        }
        return instance;
    }
}
```

4. 懒汉方式4：静态内部类

```java
public class Singleton {
​
    //私有构造方法
    private Singleton() {}
​
    private static class SingletonHolder {
        private static final Singleton INSTANCE = new Singleton();
    }
​
    //对外提供静态方法获取该对象
    public static Singleton getInstance() {
        return SingletonHolder.INSTANCE;
    }
}
```

第一次加载Singleton类时不会去初始化INSTANCE，只有第一次调用getInstance，虚拟机加载SingletonHolder并初始化INSTANCE，这样不仅确保线程安全，也保证Singleton类的唯一性。

注：开源项目中比较常用的一种单例模式。在没有加任何锁的情况下，保证了多线程下的安全，并且没有任何性能影响和空间的浪费。

