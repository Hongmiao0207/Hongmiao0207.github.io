# 了解Java Stream流操作


## Static Class 静态类

可以在一个类里面定义静态类，比如，内部类 (nested class)。把nested class封闭起来的叫外部类。只有内部类才可以static。

### 静态内部类和非静态内部类的不同

1. 内部静态类不需要有指向外部类的引用，但非静态内部类需要持有对外部类的引用。
2. 非静态内部类能够访问外部类的静态和非静态成员，静态类不能够访问外部类的非静态成员，智能访问外部类的静态成员。
3. 非静态内部类不能够脱离外部类实体被创建，它可以访问外部类的数据和方法，因为它就在外部类里面

### The difference between Comparable and Comparator

Comparable是一个对像本身就已经支持自比较所需要实现的接口（如 String、Integer自己就可以完成比较大小操作，已经实现了该接口）。

Comparable接口只提供了   int   compareTo(T   o)方法，也就是说假如我定义了一个Person类，这个类实现了   Comparable接口，那么当我实例化Person类的person1后，我想比较person1和一个现有的Person对象person2的大小时，我就可以这样来调用：person1.comparTo(person2),通过返回值就可以判断了；而此时如果你定义了一个   PersonComparator（实现了Comparator接口）的话，那你就可以这样：PersonComparator   comparator=   new   PersonComparator();
comparator.compare(person1,person2)；。

Comparator是一个专用的比较器，当前对象不支持自比较或者自比较函数不能够满足要求时，可以写一个比较器来完成对象之间的大小比较。

Comparator定义了俩个方法，分别是 int compare 和 boolean equals，用于比较两个Comparator是否相等。

注意：有时在实现Comparator接口时，并没有实现equals方法，可程序并没有报错，原因是实现该接口的类也是Object类的子类，而Object类已经实现了equals方法.

### Iterator 迭代器

迭代器是一种设计模式，是一个对象，遍历选择序列中的元素。

.next() 向下移动指针，返回下一个元素，如果没有元素，就报异常。（当调用.next()，会自动调用.remove()，也就是说迭代器遍历一次，该位置的元素会被清除）

.hasNext() 返回True表示有下个元素，false表示没有元素了，此时只判断下一个元素的有无，并不移动指针。

.remove() 删除的是指针指向的元素，当前指针没有元素，那么会抛异常。

#### 外部迭代

```java
public static void traditionalIterator(ArrayList<Artist> artList) {
    int count = 0;
    Iterator<Artist> iterator = artList.iterator();
    System.out.println(iterator.hasNext());
    while (iterator.hasNext()) {
        iterator.next();
        count++;
    }
    System.out.println("外部迭代器："+count);
}
```

使用迭代器的方法调用流程：4次

1. hasNext() 查询 +1
2. hasNext() 返回查询值 +1
3. next() 移动指针 +1
4. next() 返回当前指针的元素值 +1

#### 内部迭代

```java
public static void streamIterator(ArrayList<Artist> artList) {
    long count = artList.stream().
            filter(artist -> 0 != artist.getAge())
            .count();
    System.out.println("内部迭代器："+count);
}
```

使用stream类库后的方法调用流程：2次

1. 构建操作
2. 返回结果

### 函数式编程 Stream 流

```java
artList.stream().filter(artist -> 0 != artist.getAge());

long count = artList.stream().
                filter(artist -> 0 != artist.getAge())
                .count();
```

#### 只过滤不记数

像 filter 这样只描述Stream，最终不产生新集合的方法叫做**惰性求值方法**

而像 count 这样最终会从 Stream 产生值的方法叫做 **及早求值方法**。

#### 由于使用了惰性求值，没有输出艺术家名字

```java
// 它本身有返回值，是Stream类型，因为不需要进一步操作返回值，所以就写成这样
artList.stream().filter(artist -> {
   System.out.println(artist.getName());
   return artist.getName().equals("阿凡达");
});
```

#### 输出艺术家名字

```java
long getName = artList.stream().filter(artist -> {
   System.out.println(artist.getName());
   return artist.getName().equals("阿凡达");
}).count(); 
```

输出结果：将集合中所有的元素都打印出来，和return返回什么无关，只是打印getName。

```java
阿凡达
奇异博士
```

前面两个区别，前者是惰性求值（个人理解是没有返回），因此不会打印。而同样的语句在后者，因为加入了一个终止操作的流，就比如记数操作，名字就被会打印。（**不懂其原理**）

#### 解答

判断一个操作是惰性求值还是及早求值的方法，关注其返回值

1. 返回值是Stream就是惰性操作。（不会展示打印内容）
2. 返回值是其它类型或者空，就是及早操作。（会展示打印内容）

```java
Stream<Artist> getNameStream = artList.stream().filter(artist -> {
   System.out.println(artist.getName());
   return artist.getName().equals("阿凡达");
});

long getName = artList.stream().filter(artist -> {
   System.out.println(artist.getName());
   return artist.getName().equals("阿凡达");
}).count();
```

使用这些操作的理想方式是形成一个惰性求值的链，最后用一个及早求值的操作返回想要的结果，这正是它的合理之处。

整个过程和建造者模式有共同之处。建造者模式使用一系列操作设置属性和配置，最后调用一个build方法，这时，对象才被真正创建。

#### 为什么要区分惰性求值和及早求值

只有在对需要什么样的结果和操作有了了解后，才能更有效率地进行计算。

比如，如果要找到大于10的第一个数，并不需要和所与元素进行不交，只需要找到第一个匹配的元素即可。这意味着可以在集合类上级联多种操作，但迭代只需要一次。

### 常用的流操作

#### collect(toList())

collect(toList()) 方法由Stream里的值生成要给列表，是一个及早求值操作。

Stream的of方法使用一组初始值生成新的Stream。

```java
// 创建List
List<String> list = Stream.of("a", "b", "c", "d").collect(Collectors.toList());

// 方式2 该方法应该是上面的进化版，更简洁
List<String> list = Stream.of("a", "b", "c", "d").toList();
```

首先由列表生成一个Stream，然后进行一个Stream上的操作，继而是collect操作，由Stream声称列表。

#### map

如果有一个函数可以将一种类型的值转换成另外一种类型，map操作就可以使用该函数，讲一个流中的值转成一个新的流。

比如，将一组字符串转换成大写形式，传统方式是在循环中对每个字符串调用 toUppercase方法，然后将得到的结果加入到一个新的列表。

传统小写变大写的方法

```java
ArrayList<String> collected = new ArrayList<>();
for (String str : asList("a","b","hello")) {
    String s = str.toUpperCase();
    collected.add(s);
}
···

Stream小写变大写
```java
List<String> collect = Stream.of("a", "b", "hello")
        .map(str -> str.toUpperCase())
        .collect(Collectors.toList());

// 简洁版
List<String> collect = Stream.of("a", "b", "hello")
        .map(String::toUpperCase).toList();
```

传给map的Lambda表达式只接受一个String类型的参数，返回一个新的String。参数和返回值不必属于同一种类型，但是Lambda表达式必须是Function接口的一个实例，Function接口是只包含一个参数的普通函数接口。

#### filter

遍历数据并检查其中的元素时，可以尝试使用filter。主要作用就是筛选出有用的信息。

例：找出一组字符串中以数字开头的字符串，

传统方法：

```java
ArrayList<String> list = new ArrayList<>();
for (String str : asList("a", "1a", "a1", "2b")) {
    if (Character.isDigit(str.charAt(0))) {
        list.add(str);
    }
}
```

Stream流方式：

```java
List<String> list = Stream.of("a", "1a", "a1", "2b")
        .filter(s -> Character.isDigit(s.charAt(0)))
        .toList();
```

和map很像，filter接受要给函数作为参数，该函数用Lambda表达式表示。如果for循环中的if条件芋圆可以被filter代替。他们的返回值都是true或false，来过滤Stream中符合条件的。

#### flatMap

flatMap可以将一个类中不同属性但是同类型的元素装载到一个Stream中，但是map做不到

```java
list.stream()
        .flatMap(artist -> Stream.of(art.getName(), art.getNationality()))
        .toList();

list.stream()
        .map(artist -> artist.getName())
        .toList();
```

这样返回的集合就是说我们想要的艺术家的年龄的集合。

##### flatMap 和 map

map可用一个新的值代替Stream中的值。而flatMap是生成新的Stream对象取代之前，或者是之前都n个流对象，flatMap将他们整合后变成一个。

```java
List<Integer> together = Stream.of(asList(1, 2), asList(3, 4))
                .flatMap(numbers -> numbers.stream())
                .toList();
// Idea推荐写法
List<Integer> together = Stream.of(asList(1, 2), asList(3, 4))
                .flatMap(Collection::stream)
                .toList();
```

#### max & min

求最值

```java
// 书中给出的写法
Artist minArtistAge1 = list.stream()
        .min(Comparator.comparing(artist -> artist.getAge()))
        .get();

// Idea推荐写法
Artist minArtistAge1 = list.stream()
        .min(Comparator.comparing(Artist::getAge))
        .get();

Artist maxArtistAge = list.stream()
        .max(Comparator.comparing(Artist::getAge))
        .get();

Optional<Artist> max = list.stream()
        .max(Comparator.comparing(Artist::getAge));
```

排序的时候需要用到 Comparator 比较器对象。Java8中提供了新的静态方法 comparing()，使用它可以方便地实现一个比较器。之前需要比较两个对象的某个属性值。现在一个存取方法就够了。

* 需要深入研究comparing方法，该方法接受一个函数并返回另一个函数，该方法本该早已加入Java标准库，但是由于匿名内部类可读性差且书写冗长，一直未实现。现在在Lambda表达式，可以简单的完成。

调用空Stream的max/min方法，返回的是Optional对象，它代表一个可能存在也可能不存在的值。它里面的实际的值就是我们要得到的对象是在 .max().get() / .min()/get() 得到的。

#### 通用模式

max和min方法都属于更通用的一种编程模式

使用for循环重写上面的代码：

```java
List<Artist> list = List.of(new Artist(11, "China", "Sam"),
                new Artist(9, "UK", "Die"),
                new Artist(22, "US", "Daming"));

Artist artist = list.get(0);
for(Artist obj : list){
    if(obj.getAge() < artist.getAge()){
        artist = obj;
    }
}
```

#### reduce 聚合方法

加减乘除都行。把Stream的所有的元素按照聚合函数聚合成一个结果

传统reduce模式：

```java
List<Integer> numbers = List.of(1, 2, 3, 4);
Integer accumulator = 0;
for(Integer element : numbers){
    accumulator = element + accumulator;
}
```

Stream中reduce模式：

```java
Integer count = Stream.of(1, 2, 3, 4)
        .reduce(0, (acc, element) -> acc + element);

// idea推荐写法，0表示和列表中进行计算的数字，Integer表示类型，sum表示计算方式
Integer count = Stream.of(1, 2, 3, 4)
        .reduce(0, Integer::sum);

展开reduce操作：
```java
// 在这定义两个元素是相加的
BinaryOperator<Integer> accumulator = (acc, element) -> acc + element;
// 或者
BinaryOperator<Integer> accumulator1 = Integer::sum;

// 这里给accumulator添加元素，apply里就是两个元素
// 这两个元素会按照accumulator中的运算法则进行运算
int count = accumulator1.apply(
        accumulator1.apply(
                accumulator1.apply(
                        accumulator1.apply(0, 1),
                        2),
                3),
        4);
```

**Reduce参数**：

```java
Performs a reduction on the elements of this stream, using the provided identity value and an associative accumulation function, and returns the reduced value. This is equivalent to:

 T result = identity;  
 for (T element : this stream)      
        result = accumulator.apply(result, element)  
 return result;

but is not constrained to execute sequentially.

The identity value must be an identity for the accumulator function. This means that for all t, accumulator.apply(identity, t) is equal to t. The accumulator function must be an associative function.

This is a terminal operation.

Params: 
        identity – the identity value for the accumulating function 
        accumulator – an associative, non-interfering, stateless function for combining two values

Returns: the result of the reduction

API Note:
Sum, min, max, average, and string concatenation are all special cases of reduction. Summing a stream of numbers can be expressed as:
 Integer sum = integers.reduce(0, (a, b) -> a+b);
or:
 Integer sum = integers.reduce(0, Integer::sum);
While this may seem a more roundabout way to perform an aggregation compared to simply mutating a running total in a loop, reduction operations parallelize more gracefully, without needing additional synchronization and with greatly reduced risk of data races.

T reduce(T identity, BinaryOperator<T> accumulator);
```

* 需要传入两个参数，其中第一个参数，identity，表示的是初始值，以它为基础开始运算。
* 第二个参数 accumulator，类型是 BinaryOperator\<T> ，它继承了 Bifunction<T,T,T>，而它当中有一个 apply的方法。他们每次操作两个元素。

#### 整合操作

Stream接口的方法很多，让人难以选择，本节举例说明如何将问题分解为简单的stream操作。

问题1：找到某张专辑上所有乐队的国籍。艺术家列表中既有个人也有乐队。

将问题分解为如此几个步骤：

1. 找出专辑上的所有表演者 ——> 目标对象有个getMusicians方法，返回一个Stream对象，包含所有的表演者
2. 分辨哪些是乐队 ——> filter方法对表演者过滤，只保留乐队，具体方法可以筛选以The为开头的（打个比方）
3. 找出每个乐队的国籍 ——> map方法将乐队映射为其所属国家
4. 将找到的的国籍放入一个集合 ——> .toList)()方法将国籍放入一个集合

```java
Set<String> getNationalities = album.getMusicians()
        .filter(artist -> artist.getName().startsWith("The"))
        .map(artist -> artist.getNationality())
        .toList();
```

这个例子将链式操作展现的淋漓尽致，调用filter, map都返回Stream对象，因此属于惰性求值，而toList()属于及早求值。map方法接受一个Lambda表达式，使用该Lambda表达式对Stream上的每一个元素做映射，形成一个新的Stream。

思考，真的需要对外暴露一个List或Set对象么？通过Stream暴露集合的最大优点在于，它很好地封装了内部实现的诗句结构。仅暴露一个Stream接口，在实际操作中用户无论如何操作，都不会影响到内部的List或Set。

### 重构遗留代码

传统的代码

```java
public Set<String> findLongTracls(LList<Album> albums){
        Set<String> trackNames = new HashSet<>();
        for(Album album : albums){
                for(Track track : album.getTrackList()){
                        if(track.getLength() > 60){
                                String name = track.getName();
                                trackNames.add(name);
                        }
                }
        }
}
```

Stream重构

```java
public Set<String> findLongTracks(List<Album> albums){
        Set<String> trackNames = new HashSet<>();
        albums.forEach(album -> {
                        album.getTracks()
                                .forEach(track -> {
                                        if(track.getLength() > 60){
                                                String name = track.getName();
                                                trackNames.add(name);
                                        }
                                });
                });
        return trackNames;
}
```

对比发现重构后的代码的可读性比之前的代码还差。

分析：最内层的foreach的作用：

1. 找出长度大于1的曲目
2. 得到符合条件的曲目名称
3. 将其添加到Set

这意味着需要三项Stream操作，找到满足条件的纠结过是 filter功能，得到曲目名称可以用map，最终使用forEach将曲目添加到filter功能。

重构2：

```java
public Set<String> findLongTracks(List<Album> albums){
        Set<String> trackNames = new HashSet<>();
        albums.forEach(album -> {
                album.getTracks()
                .filter(track -> track.getLength() > 60)
                .map(track -> track.getName())
                .forEach(name -> tracksNames.add(name))
        });
        return trackNames;
}
```

代码看着还是冗余，继续改：

```java
public Set<String> findLongTracks(List<Album> albums){
        Set<String> trackNames = new HashSet<>();
        albums.stream()
                .flatMap(album -> album.getTracks()))
                .filter(track -> track.getLength() > 60)
                .map(track -> track.getName))
                .forEach(name -> trackNames.add(name));
        return trackNames;
}
```

上面代码已经替换了两个嵌套的for循环，看起来清晰很多，但是并未及家属，仍需要手动创建一个Set对象并将其加入到其中，下面将通过Stream完成。

```java
public Set<String> findLongTracks(List<Album> albums){
        return albums.stream()
                .flatMap(album -> album.getTracks())
                .filter(track -> track.getLength() > 60)
                .map(track -> track.getName)
                .collect(Collectors.toSet());
}
```

上述步骤中没有提到一个重点，就是每一步代码都要编写单元测试。

### 多次调用流操作

虽然可以选择每一步强制对函数求值，而不是所有的方法调用连接在一起，但是最好不要这么做。(能写在一起就不要分开写几行)。

### 正确使用Lambda表达式

明确了要达成什么转化，而不是说明如何转化。其另一层含义在于写出的函数没有副作用。没有副作用的函数不会改变程序或外界的状态。比如，向控制台输出了信息，就是一个可观测到的副作用。

下面代码有无副作用？

```java
private ActionEvent lastEvent;

private void registerHandler(){
        button.addActionListener((ActionEvent event) -> {
                this.lastEvent = event;
        })
}
```

这里将参数event保存至成员变量lastEvent。给变量赋值也是一种副作用，而且更难察觉。的确是改变了程序的状态。

程序鼓励用户使用Lambda表达式获取值而不是变量。获取值使用户更容易写出没有副作用的代码。

无论何时，将Lambda表达式传给Stream上的高阶函数，都应该尽量避免副作用，除了forEach，因为它是一个终结方法。

## 类库

### 在代码中使用Lambda表达式

#### 降低日志性能开销

#### 基本类型

基本类型对应着装箱类型，int - Integer。后者是java的普通类，是对基本类的封装。

Java泛型是基于对泛型参数类型的擦除。只有装箱类型才能作为泛型参数，基本类型不可以。

但是装箱类型是对象，这表示在内存中存在额外的开销。比如，整型的在内存中占用4字节，整形对象却占16字节。数组这一情况更严重，整型对象数组中，每个元素都是内存的一个指针，只想堆中某个对象。最坏情况，同样的数组，Integer[] 要比 int[] 多占用6倍内存。

而二者互相转换的过程称为装箱和拆箱，过程中都需要额外的计算开销，会影响程序运行速度。

Stream类中某些方法对基本类型和装箱类型做了区分，进而减小些性能开销。

命名规范，返回类型为基本类型，则在基本类型前加To。参数是基本型，则就直接基本类型即可。比如：ToLongFunction 和 LongFunction。

这些处理基本类型线管的Stream和一半的Stream是有区别的，比如LongStream。而这类特殊的Stream中的map方法的实现方式也不同，它接受的是一个LongUnaryOperator函数，将一个长整型映射成另一个长整型值。

```java
public static void example1(){
   List<Artist> artists = List.of(new Artist(11, "China", "Sam"),
          new Artist(9, "UK", "Die"),
          new Artist(22, "US", "Daming"));
   IntSummaryStatistics intSummaryStatistics = artists.stream()
          .mapToInt(Artist::getAge)
          .summaryStatistics();
   System.out.printf("Max: %d, Min: %d. Ave: %f, Sum: %d",
          intSummaryStatistics.getMax(),
          intSummaryStatistics.getMin(),
          intSummaryStatistics.getAverage(),
          intSummaryStatistics.getSum());
}
```

#### 重载解析

Lambda表达式作为参数时，其类型由它的目标类型推导得出，遵循如下规则：

* 如果只有一个可能的目标类型，由相应函数接口里的参数类型推导得出；
* 如果有多个可能的目标类型，由最具体的类型推导得出；
* 如果有多个可能的目标类型且最具体的类型不明确，则需要认为指定类型。

#### @FunctionalInterface

实际上，每个用作函数接口的接口都应该添加这个注释。该注释糊强制javac检查一个接口是否符合函数接口的标准。

如果该注释添加给一个枚举类型、类或另一个注释，或者接口包含不止一个抽象方法，javac就会报错。

重构代码时，使用它很容易发现问题。

#### 新的语言特性：默认方法

为了保证二进制接口的兼容性而出现的。

新关键字: default，这个关键字告诉javac，用户真正需要的是为接口添加一个新方法，从而进行区分。

三定律：

* 类胜于接口。如果在继承链中有方法体或抽象的方法声明，那么就可以忽略接口中定义的方法。（让代码向后兼容）
* 子类胜于父类。如果一个接口继承了另一个接口，且两个接口都定义了一个默认方法，那么子类定义的方法胜出。
* 如果上面两个规则不适用，子类要么需要实现该方法，要么将该方法声明为抽象方法。

#### Optional

reduce方法有一个重点，它有两种形式：

1. 需要有一个初始值，上面有例子。
2. 不需要有初始值，reduce第一步使用Stream中的前两个元素，此时返回一个Optinal对象。

Optional用来替换null值，因为null最大的问题在于NullPointerException。

Optional两个目的，鼓励程序员适时检查变量是否为空，避免代码缺陷；二它将是一个类的API中可能为空的值文档化，这比阅读实现代码要简单的多。

使用方式，在调用get之前，先使用 isPresent()检查Optional对象是否有值，或者用 orElse()。

## 高级集合类和收集器

### 方法引用

Lambda通常表示方式：

```java
artist -> artist.getName()
```

方法引用是一种简写语法：

```java
Artist::getName
```

标准语法为：Classname::methodName

注意：虽然是方法，但是不需要在后面加括号，因为这不属于调用，这是方法引用，当需要的时候，系统会自动调用。

#### 构造函数的缩写形式

##### 创建对象

Lambda方式：

```java
(name,nationality) -> new Artist(name,nationality)
```

方法引用

```java
Artist::new
```

##### 创建数组

```java
String[]::new
```

### 元素顺序

集合类的内容是流中的元素以何种顺序排列，比如顺序list；无序hashset。增加了流操作后，顺序问题变得更加复杂。

直观上，流是有序的，因为流中的元素都是按照顺序处理的，这叫做出现顺序。它依赖于数据源和对流的操作。

在有序集合中创建流，流中元素就是有序的。集合本身是无序的，生成的流则也是。

```java
List<Integer> numbers = asList(1,2,3,4);
List<Integer> sameOder = numbers.stream().toList();

Set<Integer> numbers = new HashSet<>(asList(4,3,2,1));
List<Integer> sameOder = numbers.stream().toList();
```

无序集合生成出现顺序

```Java
Set<Integer> numbers = newHashSet<>(asList(4,3,2,1));
// 流中选择排序
List<Integer> sameOder = numbers.stream().sorted().toList();
```

### 使用收集器

从java.until.stream.Collectors类中导入

#### 转换成其他集合

收集器生成其他集合，比如 toList、toSet、toCollection。

通常情况创建集合需要指定具体类型：

```java
List<String> strList = new ArrayList<>();
```

而toList或toSet不需要指定具体类型。Stream会组动挑选合适的类型。

指定集合类型：

```java
stream.collect(toCollection(TreeSet::new));
```

#### 转换成值

maxBy 和 minBy 允许用户按照特定的顺序生成值。

```java
public Optional<Artist> biggestGroup(Stream<Artist> artists){
        Function<Artist, Long> getCount = artist -> artist.getMembers().count();
        return artists.collect(maxBy(comparing(getCount)));

        // 也可以是
        return artists.max(comparing(getCount));
} 
```

求平均值

```java
Double collect = artList.stream()
                .collect(averagingInt(Artist::getAge));
```

#### 数据分块

将其分解成两个集合。假设有一个A流，将其分为B流和C流，可以通过两次过滤操作得到两种流。但问题是，为了执行两次过滤操作就需要两个流，其次如果过滤操作复杂，两次的操作就会导致代码的冗余。

解决办法：收集器 partitioningBy，它接受一个流，并将其分为两部分。它使用 Predicate对象判断一个元素应该属于哪个部分，并根据boolean返回一个Map到列表。因此，对于true List中的元素，Predicate返回true，对于其他list中的元素，返回false。

```java
Map<Boolean, List<Artist>> booleanListMap = artList.stream()
                                                .collect(partitioningBy(artist -> artist.getAge() < 13));

// 如果方法自带判断属性也可以直接用方法引用
Map<Boolean, List<Artist>> booleanListMap = artList.stream()
                                                .collect(partitioningBy(Artist::isSolo));

```

#### 数据分组

它是更自然的分割数据操作，与数据分块不同（将数据分成True、False两部分不同），它可以使用任意值对数据分组。groupingBy()

```java
Map<Integer, List<Artist>> collect = artList.stream().collect(groupingBy(Artist::getAge));
```

#### 字符串

最后生成一个字符串。

传统方式：

```java
StringBuilder builder = new StringBuilder("[");
for(Artist artist : artists){
        if(builder.length() > 1){
                builder.append(",");
        }

        String name = artist.getName();
        builder.append(name);
}
builder.append("]");
String result = builder.toString();
```

Stream方式：

```java
String result = 
        artists.stream()
                .map(Artist::getName)
                .collect(Collectors.joining(",","[","]"));
```

#### 组合收集器

在数据分组上进行改进，不需要返回两个对象的列表，而是返回对象中的某个属性。

比如，计算一个艺术家的专辑数量：

```java
public Map<Artist, Long> numberOfAlbums(Stream<Album> albums){
        return albums.collect(groupingBy(album -> album.getMainMusician(),
                                         counting));
}
```

重点是另一个收集器，counting，使用它就可以对专辑记数了。

比如，这次不想得到一组专辑，而是要得到专辑名：

```java
public Map<Artist, Long> numberOfAlbums(Stream<Album> albums){
        return albums.collect(groupingBy(Album::getMainMusician,
                                        mapping(Album::getName, toList())));
}
```

mapping收集器和map方法一样，接受一个Function对象作为参数。它允许在收集器的容器上执行类似map操作，但是需要致命使用什么样的集合类存储结果。

不管是mapping还是joining都叫做下游收集器。收集器是生成最终结果的一剂配方，下游收集器则是生成部分结果的配方，主收集器中会用到下游收集器。

#### 重构和定制收集器

传统连接字符串

```java
StringBuilder builder = new StringBuilder("[");
for(Artist artist : artists){
   if(builder.length() > 1){
       builder.append(",");
   }
   builder.append(artist.getName());
}
String str = builder.toString();
```

map操作：

```java
StringBuilder builder = new StringBuilder("[");
artists.stream()
       .map(Artist::getName)
       .forEach(name -> {
           if (builder.length() > 1)
               builder.append(",");
       });
builder.append("]");
String str = builder.toString();
```

感觉和传统方式没有什么优势，同样都需要大量的代码，可读性依旧不高。

使用reduce操作：

```java
StringBuilder reduce = artists.stream()
       .map(Artist::getName)
       .reduce(new StringBuilder(), (builder, name) -> {
           if (builder.length() > 1)
               builder.append(",");
           builder.append(name);
           return builder;
       }, (left, right) -> left.append(right)); // 这一步也可以写成 , StringBuilder::append
reduce.append("]");
String srt = reduce.toString();
```

这么一看，更复杂，可读性更差。

重构：

```java
 // 自定义类
StringCombiner combined = artists.stream()
       .map(Artist::getName)
       .reduce(new StringCombiner(",","[","]"),
               StringCombiner::add,
               StringCombiner::merge);
String str = combined.toString();
```

看着代码简洁了很多，但是背后的工作是一样的，只不过是将之前繁多的步骤，封装在一个StringCombiner类中。

StringCombiner.add()方法

```java
public StringCombiner add(String element){
     // 如果是在初始位置，则添加前缀
     if(areStart()){
             builder.append(prefix);
     // 否则添加 界定符（delimit）
     }else{
             builder.append(delim);
     }
     // 正常添加元素
     builder.append(element);
     return this;
}
```

add方法内其实调用的是一个StringBuilder对象。

StringCombiner.merge()方法

```java
public StringCombiner merge(StringCombiner other){
     builder.append(other.builder);
     return this;
}
```

在重构部分，将reduce操作重构为一个收集器，其实还是有些冗余。既然怎么都要创建一个自定义类，为什么不直接创建一个收集器就叫StringCollector。

重构2：

```java
String str = artists.stream()
                     .map(Artist::getName)
                     .collect(new StringCollector(",", "[", "]"));
```

将所有的对字符串的操作代理交给了定制的收集器StringCollector，应用程序就不需要再关注其内部任何细节。它和框架中其他的Collector对象用起来应该都是一样的。

要实现Collector接口，由于Collector接口支持泛型，因此先得确定一些具体的类型：

* 待收集元素的类型，这里是String；
* 累加器得类型StringCombiner；
* 最终结果的类型，依然是String；

```java
public class StringCollector implements Collector<String, StringCombiner, String>{}
```

一个收集器由四部分组成，首先是 Supplier，这是一个方法工厂，用来创建容器。在本地例中就是 StringCombiner。和reduce操作中第一个参数类似，它是后续操作的初值，也是格式。

实现了Collector接口就要重写几个方法：supplier、accumulator、combiner、finisher、characteristics。

Supplier是创建容器的工厂

```java
public Supplier<StringCombiner> supplier(){
    return() -> new StringCombiner(delim, prefix, suffix);
}
```

**accumulator是结合之前操作的结果和当前值，生成新的值**，它的作用和重构中reduce操作的第二个参数一样。

这里accumulator用来将流中的值叠加入容器中：

```java
public BiConsumer<StringCombiner, String> accumulator() {
    return StringCombiner::add;
}
```

其中，BiConsumer<T, U>表示接受两个参数并不返回结果的操作。

**combiner就是合并操作**，等同于reduce的第三个操作，也就是merge：

```java
public BinaryOperator<StringCombiner> combiner() {
    return StringCombiner::merge;
}
```

其中，BinaryOperator\<T> 表示根据两个同类型的数据进行运算/操作，产生一个和他们同行类型的结果。BinaryOperator是BiFuntion的特殊化。

**finisher操作**就是将操作类型转换成成特定类型并返回：本操作最后返回的是字符串

```java
public Function<StringCombiner, String> finisher() {
    return StringCombiner::toString;
}
```

其中 Function<T, R>表示接受一个参数T，返回一个结果R。

还有一个收集器Collector的重写方法，characteristics，它表示特征，它是一组描述收集器的对象。

上面的StringCominer的功能已经在在java.until.StringJoiner类中实现。

#### 对收集器的归一化处理

Collectors.reducing()为流上的归一操作提供了统一实现。

```java
String resut = 
        artists.stream()
                .map(Artist::getName)
                .collect(reducing(
                        new StringCombiner(",","[","]"),
                        name -> new StringCombiner(",","[","]").add(name),
                        StringCombiner::merge))
                .toString();
```

或者通过两个map达到一样的效果

它是将reducing中的第二步操作放到了map中

```java
String str = artists.stream()
                .map(Artist::getName)
                .map(name -> new StringCombiner(",", "[", "]").add(name))
                .reduce(new StringCombiner(",", "[", "]"), StringCombiner::merge)
                .toString();
```

对比，groupingBy 和 groupingByConcurrent，前面的是非并发的线程不安全的，后面是并发的同样线程不安全。

#### Map迭代方式

传统方式：

```java
Map<Artist, Integer> countOfAlbums = new HashMap<>();
for(Map.ENtry<Artist, List<Album>> entry: albumsByArtist.entrySet()){
        Artist artist = entry.getKey();
        List<Album> albums = entry.getValue();
        countOfAlbums.put(artist, albums.size());
}
```

内部迭代器：

```java
Map<Artist, Integer> countOfAlbums = new HashMap<>();
albumsByArtists.forEach((artist, albums) -> {
        countOfAlbums.put(artist, albums.size());
});
```

### 补充

#### boxed()

IntSream是int类型的流，Stream\<Integer> 是Integer类型的流；而 boxed()值得就是对基本类型的装箱。

#### mapToInt()

就是将参数转成IntStream。然后接着 .toArray()就是转成数组。

## 数据并行化

赘述为什么需要并行化和什么时候会带来性能的提升

### 并行和并发

并发：一个时间段内处理两个任务，他们之间还是有先后顺序（单核）。

并行：两个任务同时执行（多核cpu）。并行化，是指为缩短任务执行时间，将一个任务分解成几部分，然后并行执行。举例，很多马来拉车，省时省力。相应地，cpu承载的工作量比顺序执行要大。

**数据并行化**，将数据分成块，每块数据分配单独的处理单元。举例，原来一个马车拉的货，现在分给另一个马车，一边拉一半。

在处理大量数据上，数据并行化管用。

### 并行化的重要性

以前计算能力依赖单核cpu的时钟频率的提升，比如5mhz到60mhz。现在单核cpu的时钟频率提升不大，大家都转向了多核处理器，因此，想提升计算能力，就要将任务分解交个各个核同时处理，拆分的越多分配的核数越多，处理时间越迅速。

### 并行化流操作

并行化操作流只需要改变一个方法调用，将 .stream()替换成 .parallelStream()。

思考，并行化运行基于流的代码是否比串行化运行更快？影响因素有很多，比如输入流的大小，假设样本数量为10，那么串行肯定优于并行；样本数量10k，二者速度可能相仿；样本数量千万级或者远远更大，此时，并行优于串行。

### 模拟系统

搭建一个简易的模拟系统来理解摇色子，蒙特卡洛模拟法，它会重复相同的模拟很多次，每次都随机生成种子，结果被记录下来。汇总得到一个对系统的全面模拟。

### 限制

流框架会自己处理同步操作，所以要避免给数据结构加锁，否则可能会自找麻烦。

除了parallel方法还有一个叫sequential的方法，它们分别是串行和并行。如果同时调用两个方法，最后调用的有用。

### 性能

影响性能的因素：

* 数据大小，数据足够大，并行处理才有意义。
* 源数据结构，通常是集合。
* 装箱，处理基本类比装箱快。
* 核的数量，核数越多，潜在性能提升越大。
* 单元处理开销，花在流中每个元素身上的时间越长，并行操作带来的性能提升越明显。（就是单个元素越复杂，并行效果越好）。

在底层，并行流还是沿用了 fork/join框架，fork递归式分解问题，然后每段并行执行，最终由join合并结果。

将数据结构分类：

* 性能好：ArrayList、数组、IntStream.range，这些数据结构支持随机读取，可以很轻易地被任意分解。
* 性能一般：HashSer、TreeSet，不易公平地被分解，大多数分解是可能的。
* 性能差，LinkedList、Streams.iterate、BufferedReader.lines，可能要花费O(N)甚至更高的时间复杂度来分解问题，很难分解。

初始数据结构影响巨大，样本数量足够大的时候，ArrayList比LinkedList快10倍。

### 并行化数组操作

对数组的操作都在工具类Arrays中，

parallelSetAll，更新数组元素：

```java
double[] values = new doulbe[size];
Arrays.parallelSetAll(values, i -> i);
```

parallelPrefix，对时间序列数据做累加

计算简单滑动平均数

```java
double[] sums = Arrays.of(values, values.length); // 复制一份输入数据
Arrays.parallelPrefix(sums, Double::sum); // 将数组元素相加，用sums保留求和结果。比如输入0，1，2，3，4，3.5，计算后的值为 0.0，1.0，3.0，6.0，10.0，13.5
int start = n - 1;
return IntStream.range(start, sums.length) 
                .mapToDouble(i -> {
                        double prefix = i == start ? 0 : sums[i - n];
                        return (sums[i] - prefix) / n; // 使用总和减去窗口起始值，然后再除以n得到平均值
                })
                .toArray();
```

## 测试、调试、重构

### 重构候选项

#### 进进出出、摇摇晃晃

在记录日志时，isDebugEnabled()用来检查是否启用调试级别。仔细想想，一段代码不断地查询和操作对象目的就只是为了给该对象设个值。那么在检查状态这部分不应该是个内部状态么，一查询就暴露了。

传统debug

```java
Logger logger = new Logger();
if(logger.isDebugEnabled()){
        logger.debug("Look at this: " + exoensiveOperation());
}
```

lambda debug

```java
Logger logger = new Logger();
logger.debug(() -> "Look at this: " + expensiveOperation());
```

Lambda表达式更好地面向对象编程（OOP），面向对象编程的核心之一就是封装局部状态，比如日志的级别。

传统方式做的不好，isDebugEnabled方法暴露了内部封装。使用Lambda表达式，外面的代码根本不需要检查日志级别。

#### 孤独的覆盖

假设在数据库中查找艺术家，每个线程只做一次这种查询：

```java
ThreadLocal<Album> thisAlbum = new ThreadLocal<Album>(){
        @Override protected Album initialValue(){
                return database.lookupCurrentAlbum();
        }
};
```

这段代码异味是使用继承，其目的只是为了覆盖一个方法。ThreadLocal能创建一个工厂，为每个线程最多产生一个值。确保非线程安全的类在并发环境下安全使用的一种简单方式。

Java8中，可以为工厂方法withInitial传入一个Supplier对象的实例来创建对象：

```java
ThreadLocal<Alubm> thisAlbum = ThreadLocal.withInitial(() -> database.lookupCurrentAlbum());
```

此代码优于前者是因为：

1. 任何已有Supplier\<Album>实例不需要重新封装，就可以在此使用，这鼓励了重用和组合。
2. 代码简短，不用花时间在继承的样本代码上。
3. JVM会稍加在一个类。

#### 同样的东西写两遍

Don't Repeat YourSelf, DRY模式。还有其反面 Write Everything Twice, WET模式。后者代码异味多见于重复的样本代码，难于重构。

什么时候该将WET代码Lambda化？如果有一个整体上大概相似的模式，只是行为上有所不同，就可以试着加入一个Lambda表达式。

Order类的命令式实现

```java
public long countRunningTIme(){
        long count = 0;
        for(Album album : albums){
                for(Track track : tracks){
                        count += track.getLength();
                }
        }
        return count;
}

public long countMusicians(){
        long count = 0;
        for(Album album : albums){
                count += album.getMusicianList().size();
        }
        return count;
}

public long countTracks(){
        long count = 0;
        for(Album album : albums){
                count += album.getTrackList().size();
        }
        return count;
}
```

三个方法，里面的的代码基本一样。

流重写

```java
public long countRunningTimg(){
        return albums.stream()
                .mapToLong(album -> album.getTracks()
                                        .mapToLong(track -> track.getLength())
                                        .sum())
                .sum();
}

public long countMusicians(){
        return albums.stream()
                .mapToLong(album -> album.getMusicians().count())
                .sum();
}

public long countTracks(){
        return albums.stream()
                .mapToLong(album -> album.getTracks().count())
                .sum();
}
```

这段代码仍然有重用性问题。可以实现一个函数，返回long，统计所有专辑的某些特性，还需要一个lambda表达式，告诉统计专辑上的信息。java8核心类库提供这样一个类型 ToLongFunction<>。

将专辑转换成流，将专辑映射为long，然后求和。

```java
public long countFeature(ToLongFunction<Album> function){
        return albums.stream()
                .mapToLong(function)
                .sum();
}

public long countTracks(){
        return countFeature(album -> album.getTracks().count());
}

public long countRunningTime(){
        return countFeature(album -> album.getTracks()
                                        .mapToLong(track -> track.getLength())
                                        .sum());
}

public long countMusicians(){
        return countFeature(album -> album.getMusicians().count());
}
```

### 日志和打印消息

之前我们在流中想打印是在 foreach(a -> System.out.println("例如：" + a));

这样操作的缺点是，当调试代码时，比如debug，由于流只能使用一次，在某一段打断点，如果还想继续就必须重新创建流。这个就是惰性求值。

解决方案：peek

流中有一个方法peek()可以查看每一个值，同时可以继续操作流。

```java
Set<String> nationalities = album.getMusicians()
                                .filter(artist -> artist.getName().startsWith("The"))
                                .map(artist -> artist.getNationality())
                                .peek(nation -> System.out,println("例如：" + nation))
                                .collect(Collectors.<String>toSet());
```

记录日志是peek的用途之一，可以设置断点。

## 设计和架构的原则

探索如何使用Lambda表达式实现SOLID原则。

### Lambda表达式改变了设计模式

单例模式确保只生产一个对象实例，然而它却让程序变得更脆弱，难于测试。敏捷开发的流行，让测试变得更加重要，因此单例模式变成了反模式，一种要避免的模式。

本章主要讨论如何使用Lambda让设计模式变得更好。

#### 命令者模式

命令者是一个对象，它封装了调用另一个方法的所有细节。

客户端 -创建- 发起者，发起者 -调用- 命令者，客户端 -使用- 具体命令者，具体命令者 -实现- 命令者，具体命令者 -调用- 命令接受者。

* 命令者：封装了所有调用命令执行者的信息。
* 发起者：控制一个或多个命令的顺序和执行。
* 客户端：创建具体的命令者实例。

```java
public interface Editor{
        public void save();
        public void open();
        public void close();
}

public interface Action{
        public void perform();
}

public class Save implements Action{
        private final Editor editor;
        public Save(Editor editor){
                this.editor = editor;
        }

        @Override
        public void perform(){
                editor.save();
        }
}

public class Open implements Action{
        private final Editor editor;

        public Open(Editor editor){
                this.editor = editor;
        }

        @Override
        public void perform(){
                editor.open();
        }
}

public class Macro{
        private final List<Action> actions;

        public Macro(){
                actions = new ArrayList<>();
        }

        public void record(Action action){
                actions.add(action);
        }

        public void run(){
                actions.forEach(Action::perform);
        }
}

```

不同方式的调用

```java
Macro macro = new Macro();
macro.record(new Open(editor));
...


macro.record(() -> editor.open());
...

macro.record(editor::open);
...

```

#### 策略模式

策略模式能在运行时改变软件的算法行为。一个问题可以由多种算法实现，将它们封装在一个统一的接口背后，选择哪个算法就是策略模式决定的。

文件压缩就是一个例子，它提供各种压缩方式，比如 zip、gzip等。

整体流程就是：

压缩器 调用 压缩策略，具体的压缩算法实现压缩策略。

定义压缩数据的策略借口

```java
public interface CompressionStrategy{
        public OutputStream compress(OutputStream data) throws IOException;
}
```

两个类实现该接口，zip和gzip。

使用gzip算法压缩数据

```java
public class GzipCompressionStrategy implements CompressionStrategy{

        @Override
        public OutpuStream compress(OutputStream data) throws IOException{
                return new GZIPOutputStream(data);
        }
}
```

使用zip算法压缩数据

```java
public class ZipCompressionStrategy implements CompressionStrategy{
        @Override
        public OutpuStream compress(OutputStream data) throws IOException{
                return new ZIPOutputStream(data);
        }
}
```

实现Compressor类，这就是使用策略模式的地方。它有一个compress方法，读入文件，压缩后输出。它的构造函数有一个CompressionStrategy参数，调用代码后可以在运行期使用该参数决定使用哪种压缩策略，比如，可以等待用户输入选择。

在构造类时提供压缩策略

```java
public class Compressor{
        private final CompressionStrategy strategy;

        public Compressor(CompressionStrategy strategy){
                this.strategy = strategy;
        }

        public void compress(Path inFile, File outFile) throws IOException{
                try(OutputStream outStream = new FileOutputStream(outFile)){
                        Files.copy(inFile, strategy.compress(outStream));
                }
        }
}
```

通过这种方式，只需要new一个Compressor就可以使用我们想要的策略。

使用具体策略类初始化Compressor

```java
Compressor gzipCom = new Compressor(new GzipCompressionStrategy());
gzipCom.compress(inFile, outFile);

Compressor zipCom = new Compressor(new ZipCompressionStrategy());
zipCom.compress(inFile, outFile);
```

Lambda表达式或者方法引用去掉样本代码

```java
Compressor gzipCom = new Compressor(GzipCompressionStrategy::new);
gzipCom.compress(inFile, outFile);

Compressor zipCom = new Compressor(ZipCompressionStrategy::new);
zipCom.compress(inFile, outFile);
```

#### 观察者模式

被观察者持有一个观察者列表，当被观察者的状态发生改变时，会通知观察者。这种模式被大量应用于基于MVC的GUI工具中，以此让模型状态发生变化时，自动刷新视图模块，达到二者解耦。

比如，观察的是月球，NASA和Aliens都对月球感兴趣。

观察登陆到月球组织的接口

```java
public interface LandingObserver{
        public void observeLanding(String name);
}
```

Moon类

```java
public class Moon{
        private final List<LaningObserver> observers = new ArfrayList<>();

        public void land(String name){
                for(LandingObserver observer : observers){
                        observer.observeLanding(name);
                }
        }

        public void startSpying(LandingObserver observer){
                observers.add(observer);
        }
}
```

外星人观察到人类登陆月球

```java
public class Aliens implements LandingObserver{
        @Override
        public void observeLanding(String name){
                if(name.contains("Apollo")){
                        System.out.println("They're distracted, lets invade earch!");
                }
        }
}
```

NASA观察有人登陆月球

```java
public class Nasa implements LandingObserver{
        @Override
        public void observeLanding(String name){
                if(name.contains("Apollo")){
                        System.out.println("We made it");
                }
        }
}
```

使用类的方式构建

```java
Moon moon = new Moon();
moon.startSpying(new Nasa());
moon.startSpying(new Aliens());

moon.land("An asteroid");
moon.land("Apollo 11");
```

Lambda方式构建

```java
Moon moon = new Moon();
moon.startSpying(name -> {
        if (name.contains("Apollo"))
                System.out.println("We made it");
});

moon.startSpying(name -> {
        if(name.contains("Apollo"))
                System.out.println("They're distracted, lets invade earth!");
});

moon.land("An asteroid");
moon.land("Apollo 11");
```

Lambda表达式的方式可以不用重写observeLanding里面的判断。

## New相关

创建一个List/Stream:

```java
List.of(1,2,3,4)
Stream.of(1,2,3,4,5)
```

