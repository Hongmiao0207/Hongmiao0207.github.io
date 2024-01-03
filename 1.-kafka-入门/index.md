# Kafka 入门


## 1. 概述

例子：

- 日常：
  - 前端埋点记录用户的行为数据（浏览、点赞、收藏、评论等） 。
  - 将上述存储到日志中，通过Flume对日志进行实时监控并传送给Hadoop。
- 双11：
  - Flume采集的速度跟不上流量速度，因此需要在log和hadoop中间加一个Kafka集群来进行处理（缓冲）。

### 定义

传统定义：分布式的基于发布/订阅模式的消息队列（Message Queue），主要应用与大数据实时处理领域。

- 其中，发布/订阅：消息的发布者不会将消息直接发送给特定的订阅者，而是将发布的消息分为不同的类别、订阅者只接收感兴趣的消息。

最新定义：开源的分布式事件流平台（Event Streaming Platform），主要被用于高性能数据管道、流分析、数据采集和关键任务应用。

### 技术场景

Kafka主要应用于大数据场景，JavaEE中主要采用ActiveMQ、RabbitMQ、RocketMQ。

### 应用场景

缓存/消峰（双11商城）、解耦（数据源传到多个目的地）和异步通信（不需要立即响应和处理）。

- 异步通信和同步通信对比：
  - 同步：1. 填写注册信息；2.注册信息写入到数据库；3. 调用短信发送接口；4. 发送短信；5. 页面响应注册成功。
  - 异步：1. 填写注册信息；2.注册信息写入到数据库；3. 发送短信请求写入到消息队列，通过短信服务订阅消息队列来接收消息然后发送短信；4.页面响应注册成功（此步骤发生在数据库写入成功后和3步骤属于并行非串行）。
  - 注：发送短信只是告诉用户注册成功，并不是用来做验证码验证。

### 消息队列的两种模式

点对点模式：producer -> mq -> consumer（1对1）

- producer发送信息到mq，consumer主动到mq消费信息，并给mq发送一个确认信号，mq会将确认收到的消息进行删除。

发布/订阅模式：producer ->mq's topics -> consumers

- producer发布不同的消息到mq中，mq中的消息根据topic不同而分类，consumers订阅自己感兴趣的topic，不需要确认收到，mq也不会立即删除（consumer互相独立）。

### 基础架构

1. 在一个topic中，它会把数据分割为多个 partition，每个partition存储在不同服务器中，而每个服务器中的kafka被称为broker（分区概念）。
   - 比如，producer在一个topic中的数据有90T，在一个服务器中存不下，就切割为3分，partition0、1、2，分别存储到3个服务器broker0、1、2中。而这三个组成了Kafka cluster(集群)。

2. 同样地，海量数据很难被一个consumer消费，因此在consumer端会有group概念，一个group下有多个consumer，它们分别处理同一个topic下不同partition的数据（消费者组概念，并行消费）。
   - 注：一个partition在同group中只能被一个consumer消费。

3. 为了提高可用性（防止某个partition挂掉），为每个parition增加若干副本，
    - 但是副本间会有leader和follower，producer和consumer只会对leader进行生产和消费。只有当leader挂掉后，follower中会有一个称为新的leader。

4. 另外，kafka还会有一部分数据是存储到zookeeper中，
   - 它存储着kafka哪些服务器正在工作（上线） /brokers/ids/\[0,1,2]；
   - 还会记录每一个分区中的leader相关信息 /brokers/topics/first/partitions/0/state/"leader":0,"isr":\[0,2]。

注意：4中的使用在kafka2.8.0是强制要求，后续版本配置可以不采用zookeeper。不采用zookeeper的模式叫做 kraft（去zookeeper化，大势所趋）。

## 2. 入门

### 2.1 安装部署

#### 2.1.1 集群规划

||||
|-|-|-|
|hadoop102|hadoop103|hadoop104|
|zk|zk|zk|
|kafka|kafka|kafka|

#### 2.1.2 安装

下载地址：https://kafka.apache.org/downloads

解压：

```shell
tar -zxvf kafka-3.6.0-src.tgz -C /opt/module/
```

重命名：

```shell
mv kafka-3.6.0-src/ kafka
```

其中，主要目录：

- bin，启动目录
  - kafka-consle-consumer.sh
  - kafka-consle-producer.sh
  - kafka-server-start.sh
  - kafka-server-stop.sh
- config，配置目录
  - consumer.properties
  - producer.properties
  - server.properties

server.properties

```properties
############################# Server Basics #############################

# The id of the broker. This must be set to a unique integer for each broker.
# 配置服务器id，每个id要是唯一的，因此要修改
broker.id=0


############################# Socket Server Settings #############################

# 具体看源文件


############################# Log Basics #############################

# A comma separated list of directories under which to store log files
# 存放数据的地址，不能是临时地址，会有被清除的风险
# 默认配置 log.dirs=/tmp/kafka-logs
log.dirs=/opt/module/kafka/data


############################# Zookeeper #############################

# Zookeeper connection string (see zookeeper docs for details).
# This is a comma separated host:port pairs, each corresponding to a zk
# server. e.g. "127.0.0.1:3000,127.0.0.1:3001,127.0.0.1:3002".
# You can also append an optional chroot string to the urls to specify the
# root directory for all kafka znodes.
# 默认配置 zookeeper.connect=localhost:2181
# 由于我们是集群，因此只配置本地是不行的
zookeeper.connect=kaka01:2181,kafka02:2181,kafka03:2181/kafka
# 其中hadoop10*是已经在hosts配置好的地址
```

此外，我们还要安装zookeeper

官网：https://zookeeper.apache.org/

启动命令：bin/kafka-server-start.sh -daemon config/server.properties （kafka目录下执行）

查看是否启动成功：jps

### 集群启动脚本

```shell
#!/bin/bash

case $1 in

"start") {
        for i in kafka01 kafka02 kafka03
        do
                echo ------------------------------------ kafka $i started ---------------------------------------
                ssh $i "/opt/module/kafka/bin/kafka-server-start.sh -daemon /opt/module/kafka/config/server.properties"
        done
}
;;
"stop") {
        for i in kafka01 kafka02 kafka03
        do
                echo ------------------------------------ kafka $i stoped ---------------------------------------
                ssh $i "/opt/module/kafka/bin/kafka-server-stop.sh stop"
        done
}
;;
esac
```

注：start命令失效，stop命令可用

### Topic命令

kafka分为 producer、topic、consumer，对应的脚本分别为：

- kafka-console-producer.sh
- kafka-topics.sh
- kafka-console-consumer.sh 

其中，有关topic的，我们执行 bin/kafka-topics.sh查看

```shell
--alter                                  更改分区数和副本分配

--at-min-isr-partitions                  仅显示 ISR 计数等于配置的最小值的分区

--bootstrap-server <String: server to    REQUIRED: 连接到kafka broker主机名称和端口号
  connect to>                              

--command-config <String: command        仅与--bootstrap-server一起使用，传递给Admin 
  config property file>                  Client用于描述和更改代理配置 
               
--config <String: name=value>            更新系统默认配置

--create                                 Create a new topic.

--delete                                 Delete a topic

--delete-config <String: name>           

--describe                               List details for the given topics.

--exclude-internal                       运行 list 或 describe 命令时排除内部主题

--help                                   Print usage information.

--if-exists                              

--if-not-exists                          

--list                                   List all available topics.

--partitions <Integer: # of partitions>  设置分区数

--replica-assignment <String:            A list of manual partition-to-broker
  broker_id_for_part1_replica1 :           assignments for the topic being
  broker_id_for_part1_replica2 ,           created or altered.
  broker_id_for_part2_replica1 :
  broker_id_for_part2_replica2 , ...>

--replication-factor <Integer:           设置分区副本
  replication factor>     

--topic <String: topic>                  

--topic-id <String: topic-id>            

--topics-with-overrides                  

--unavailable-partitions                 

--under-min-isr-partitions               

--under-replicated-partitions            

--version                                
```

- 查看连接的topic节点：bin/kafka-topics.sh --bootstrap-server kafka01:9092,kafka02:9092 --list (个人环境下，一个即可，即只需要kafka01:9092)

- 创建topic节点：bin/kafka-topics.sh --bootstrap-server kafka01:9092 --topic first --create --partitions 1 --replication-factor 3
  - 总的来说需要4步：
    - 指定连接的服务器名称和端口号，--bootstrap-server
    - 指定 topic（主题），--topic first
    - 创建，--create
    - 指定 paritions（分区），--paritions 1
    - 指定 replication-factor（分区副本），--replication-factor 3

创建成功显示：Created topic first.

查看节点：bin/kafka-topics.sh --bootstrap-server kafka01:9092 --list

查看节点详情：bin/kafka-topics.sh --bootstrap-server kafka01:9092 --topic first --describe

修改分区：bin/kafka-topics.sh --bootstrap-server kafka01:9092 --topic first --alter --partitions 3

- 注意，分区数的修改只能增加，不能减少。

验证就是查看节点详情

修改分区备份：不能在命令行上修改

### 生产者和生产者命令展示

- 生产者发送数据命令：bin/kafka-consle-prodcer.sh --bootstrap-server kafka01:9092 --topic first，输入命令后就可以发送消息。

- 消费者消费数据命令：bin/kafka-consle-consumer.sh --bootstrap-server kafka01:9092 --topic first，该命令可以消费到的是实时数据，对于历史数据，需要在末尾加上 --from-beginning。

## 3. 生产者

### 原理

生产者中：

1. main线程（producer）通过调用send(producerRecord)方法来发送消息

2. 发送的数据先会经过 interceptors（拦截器），此拦截器是required，不是必须（生产环境中更多的是用flume中的拦截器）

3. 紧接着数据会经过 serializer（序列化器）
   - Java中本身就序列化，为什么不走Java的序列化？
     - 原因是Java的序列化传输的数据太重，除了本身要传输的数据，它还有很大的其他辅助数据的占比，来保证安全传输等。在大数据场景，大数据量下再额外加重每个数据整体的效率就会差很多。

4. 序列化后会经过 partitioner（分区器），其作用是将数据发送到RecordAccumulator中。
   - 在RecordAccumulator中，会根据不同分区会创建对应队列（DQueue），partitioner会决定将数据发送到哪个分区下的DQueue。
   - RecordAccumulator是在内存中进行的，其默认内存大小32M，一批次的大小16K（producerBatch）。

5. RecordAccumulator缓冲队列中的数据会通过 sender线程下的sender()方法将数据准备发送到Kafka集群，其触发契机有两个：
   1. 每批次满了，即batch.size（默认情况下16k），就会启用sender来发送数据。
   2. 达到等待时间就会发送，linger.ms，如果数据未达到batch.size，sender等到时间就会发送数据，单位ms，默认值0ms。

6. sender读取了数据后，通过NetworkClient中的request来讲数据发送到Kafka集群，默认集群中每个broker节点最多缓存5个请求。

7. 第6步的发送数据，需要通过 selector打通producer和broker的链路后才能实现。

8. Kafka集群收到数据后，会有一个应答机制acks，
    - 应答0：producer发送来的数据，不需要等数据落盘应答。
    - 应答1：producer发送来的数据，leader接收到数据后应答。
    - 应答-1：producer发送来的数据，leader 和 ISR队列里面所有节点收齐数据后应答，输入-1或者all都可以（默认）。

9. 成功，Selector收到应答后，首先会清除掉队对应请求（NetworkClient中的request），并把对应的分区队列中数据清除掉（RecordAccumulator中对应的DQueue中存的数据）

10. 失败，重试，重试次数int最大值，可修改。

### 异步发送API

先设置kafka集群配置文件，并进行序列化设置

```java
private static Producer<String, String> getStringStringProducer() {
        // 0. 配置 + 连接集群
        Properties properties = new Properties();
        properties.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, "kafka01:9092,kafka02:9092");
        // 对key、value进行序列化
        properties.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
        properties.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());

        // 1. 创建生产对象
        return new KafkaProducer<>(properties);
}
```

### 普通异步发送

异步发送发生在外部数据发送到RecordAccumulator中的DQueue期间，它不会关注队列是否成功把数据发送到kafka集群，外部数据只会一批批地发送数据到队列。

```java
public static void commonAsync() {
        Producer<String, String> producer = getStringStringProducer();

        // 2. 发送数据
        for (int i = 1; i <= 10; i++) {
            producer.send(new ProducerRecord<String, String>("first", "send message by common async " + i + " times"));
        }

        // 3. 关闭资源，不关闭资源就收不到消息
        producer.close();
}
```

### 回调异步发送

在发送函数 send() 中多了一个返回函数Callback，里面有topic、partition等信息。

其中，需要重写CallBack中的 onCompletion()方法，通过RecordMetadata类型查看要返回的信息。

```java
public static void callBackAsync() {
        Producer<String, String> producer = getStringStringProducer();

        for (int i = 1; i <= 10; i++) {
            producer.send(new ProducerRecord<String, String>("first", "send message by call back async " + i + " times"), new Callback() {
                @Override
                public void onCompletion(RecordMetadata metadata, Exception e) {
                    if (e == null){
                        System.out.println(metadata.topic() + ", " + metadata.partition());
                    }
                }
            });
        }
        // 3. 关闭资源
        producer.close();
}
```

### 同步发送API

同步数据与异步数据的不同是在于，

- 异步数据发送时，不需要等待RecodrAccumulator把队列中消息发送到Kafka集群，就能发送下次的数据。
- 同步数据发送时，必须等待RecodrAccumulator把队列中消息发送到Kafka集群完成后，才能发送下次数据。

在API调用方面，和 common sync的区别就在于，它会在方法后面加上一个.get()。

```java
public static void sync() throws ExecutionException, InterruptedException {
    Producer<String, String> producer = getStringStringProducer();

    for (int i = 1; i <= 10; i++) {
        // 多了一个.get()
        producer.send(new ProducerRecord<String, String>("first", "send message by sync " + i + " times")).get();
    }

    // 3. 关闭资源
    producer.close();
}
```

### 分区

好处：

- 便于合理使用存储资源，每个 partition 在 一个 broker 上存储，可以把海量数据按照分区切割存储在多个 broker上，实现负载均衡。

- 提升并行度， producer 可以以 分区为单位 发送数据；consumer 额可以以 分为单位 消费数据。

### 分区策略

- 默认分区器 DefaultPartitioner（弃用）

```java
@Deprecated
public class DefaultPartitioner implements Partitioner {
    private final StickyPartitionCache stickyPartitionCache = new StickyPartitionCache();

    public DefaultPartitioner() {
    }

    public void configure(Map<String, ?> configs) {
    }

    public int partition(String topic, Object key, byte[] keyBytes, Object value, byte[] valueBytes, Cluster cluster) {
        return this.partition(topic, key, keyBytes, value, valueBytes, cluster, cluster.partitionsForTopic(topic).size());
    }

    public int partition(String topic, Object key, byte[] keyBytes, Object value, byte[] valueBytes, Cluster cluster, int numPartitions) {
        return keyBytes == null ? this.stickyPartitionCache.partition(topic, cluster) : BuiltInPartitioner.partitionForKey(keyBytes, numPartitions);
    }

    public void close() {
    }

    public void onNewBatch(String topic, Cluster cluster, int prevPartition) {
        this.stickyPartitionCache.nextPartition(topic, cluster, prevPartition);
    }
}
```

这段代码实现了 Partitioner 接口的 DefaultPartitioner 类，通常用在 Apache Kafka 的客户端库中。Kafka 使用分区器来确定将每条消息发送到哪个分区。

解读代码：

- 类定义：@Deprecated注解表示此类已经被弃用，未来版本可能被移除。

- 成员变量：stickyPartitionCache用于缓存分区信息，其作用是在连续的消息发送中保持对同一分区的引用，减少分区切换，从而可能提高效率。

- 主要方法：
  - partition() 重载方法，
    - 第一个采用基本参数调用第二个方法。
    - 第二个通过检查 keyBytes（消息键的字节表示）是否为空。
      - 为空，使用 stickyPartitionCache 决定分区；
      - 不为空，使用 BuiltInPartitioner.partitionForKey 方法根据键的字节数据和分区数量计算分区。
  - onNewBatch()，当开始新的消息批次时，用来更新 stickyPartitionCache 中分区信息。

也就是说，当partition指定情况下，直接将数据写入指定分区；当没有指定partition但key存在的情况下，通过key的hash值与topic的partition数 进行取余 得到partition值。如果partition值 和 key 都没有的情况下，就会通过 黏性分区（sticky partition）来随机选择一个分区，等该分区的batch满了或完成，才会再次随机一个分区进行使用。

- RoundRobinPartitioner（Kafka中唯一正在用）

```java
public class RoundRobinPartitioner implements Partitioner {
    private final ConcurrentMap<String, AtomicInteger> topicCounterMap = new ConcurrentHashMap();

    public RoundRobinPartitioner() {
    }

    public void configure(Map<String, ?> configs) {
    }

    public int partition(String topic, Object key, byte[] keyBytes, Object value, byte[] valueBytes, Cluster cluster) {
        List<PartitionInfo> partitions = cluster.partitionsForTopic(topic);
        int numPartitions = partitions.size();
        int nextValue = this.nextValue(topic);
        List<PartitionInfo> availablePartitions = cluster.availablePartitionsForTopic(topic);
        if (!availablePartitions.isEmpty()) {
            int part = Utils.toPositive(nextValue) % availablePartitions.size();
            return ((PartitionInfo)availablePartitions.get(part)).partition();
        } else {
            return Utils.toPositive(nextValue) % numPartitions;
        }
    }

    private int nextValue(String topic) {
        AtomicInteger counter = (AtomicInteger)this.topicCounterMap.computeIfAbsent(topic, (k) -> {
            return new AtomicInteger(0);
        });
        return counter.getAndIncrement();
    }

    public void close() {
    }
}
```

- 成员变量，ConcurrentMap<String, AtomicInteger> topicCounterMap，一个线程安全的映射，用于每个主题（topic）维护一个原子计数器。这个计数器用于实现轮询逻辑，以确保消息均匀地分布到各个分区。

- 主要方法：
  - partition()（核心方法），用于决定消息发送到哪个分区。
    - partitions 获取给定主题的所有分区信息。
    - numPartitions 确定总分区数。
    - nextValue 获取当前主题的下一个轮询值。
    - availablePartitions 获取可用的分区列表。
    - 如果有可用分区，该方法将基于轮询值选择一个分区；如果没有可用分区，则它将根据所有分区的数量来选择。无可用分区时的选择依据是，对所有分区的总数取模（即 Utils.toPositive(nextValue) % numPartitions）。
  - nextValue(String topic)，辅助方法，用于根据主题获取并增加计数器值。如果主题不存在于 topicCounterMap 中，则会创建一个新的计数器并初始化为 0。

两个策略对比，

- DefaultPartitioner 依赖于key的hash值来选择分区，相同 key 的所有消息将被路由到同一个分区。当没有提供 key 时，sticky 会尽可能地将连续消息发送到同一个分区，直到发生重新平衡或其他事件。确保特定用户的消息顺序或聚合特定类型的数据。

- RoundRobinPartitioner 使用轮询算法，在所有可用分区之间均匀地分配消息。确保所有分区都被平等利用，减少特定分区的过载。

### 自定义分区器

自定义一个简单的分区器。

```java
public class IndividualPartitioner implements Partitioner {
    @Override
    public int partition(String s, Object o, byte[] bytes, Object o1, byte[] bytes1, Cluster cluster) {

        // o表示key，o1是value
        String value = o1.toString();

        int partition;

        if(value.contains("individual_partitioner")){
            partition = 0;
        }else {
            partition = 1;
        }

        return partition;
    }

    @Override
    public void close() {

    }

    @Override
    public void configure(Map<String, ?> map) {

    }
}
```

测试

```java
    public static void useIndividualPartitioner(){

        // 0. 配置 + 连接集群
        Properties properties = new Properties();
        properties.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, "kafka01:9092,kafka02:9092");
        // 对key、value进行序列化
        properties.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
        properties.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());

        // 使用自定义分区器
        properties.put(ProducerConfig.PARTITIONER_CLASS_CONFIG, "org.kafka.producer.IndividualPartitioner");

        Producer<String, String> producer = new KafkaProducer<>(properties);

        producer.send(new ProducerRecord<>("first", "individual_partitioner"), new Callback() {
            @Override
            public void onCompletion(RecordMetadata recordMetadata, Exception e) {
                if(e == null){
                    System.out.println(recordMetadata.topic() + " " + recordMetadata.partition());
                }
            }
        });

        producer.close();
    }
```

### 提高吞吐量

- 修改 batch.size，批次大小，默认16k。
- 修改 linger.ms，等待时间，默认0ms，可修改为5-100ms。
- 修改 compression.tpye，压缩snappy。
- 修改 RecordAccumulator，缓冲区大小，修改为64m。

```java
public static void setProducerParameters(){
    // 配置
    Properties properties = new Properties();
    // 连接集群
    properties.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, "kafka01:9092, kafka02:9092");
    // 序列化
    properties.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
    properties.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
    // 缓冲区大小
    properties.put(ProducerConfig.BUFFER_MEMORY_CONFIG, 33554432);
    // 批次大小
    properties.put(ProducerConfig.BATCH_SIZE_CONFIG, 16384);
    // 徘徊时间
    properties.put(ProducerConfig.LINGER_MS_CONFIG, 1);
    // 压缩
    properties.put(ProducerConfig.COMPRESSION_TYPE_CONFIG, CompressionType.SNAPPY.name);

    // 创建Producer
    Producer<String, String> producer = new KafkaProducer<>(properties);

    // 发送数据
    for (int i = 0; i < 100000; i++) {
        producer.send(new ProducerRecord<>("first", "test setting parameters" + i), new Callback() {
            @Override
            public void onCompletion(RecordMetadata recordMetadata, Exception e) {
                if(e == null){
                    System.out.println("Topic: " + recordMetadata.topic() + ", Partition: " + recordMetadata.partition());
                }
            }
        });
    }
    // 关闭流
    producer.close();
}
```

### 数据可靠性

数据可靠性通过，kafka集群收到数据后返回给selector的应答acks来设置。

- 0，producer发送数据后，不需要等数据落盘就应答（可靠性最低）。
  - 数据丢失场景，producer发送数据，leader突然挂了，leader没有拿到数据，follower也没有同步到数据，导致数据丢失。

- 1，producer发送数据后，Leader收到数据后应答。
  - 数据丢失场景，producer发送数据，leader拿到数据应答后，还没同步副本就挂了，虽然follower会选举出新的leader继续接收数据，但之前没有同步到副本的数据已经丢失（因为之前挂掉的leader接收到丢失的数据已经发送应答了，那么producer就不会将之前的数据发送给新leader）。
  - 比0的优势，会存在leader消息落盘，但没来得及给应答挂了，producer就会认为发送失败，会进行重试发送。

- -1（或者 all），producer发送数据后，Leader 和 isr队列 里面所有节点收齐数据后应答，可靠性最高。
  - 比1的优势，leader拿到数据，并且所有的follower都同步完数据后，leader才会ack应答producer。
  - 可能存在的问题，如果其中某一个follower由于某种故障，不能和leader进行同步，那么leader就迟迟不能应答producer？
    - 实际上，leader维护了一个动态的 in-sync replica set（ISR），即 和leader保持同步的 follower+leader的集合（leader:0,isr:0,1,2）。
    - 如果某个follower在规定时间内未向leader已发送通信请求或同步数据，则会将该follower踢出ISR。
    - 该时间阈值由 replica.lag.time.max.ms 参数设定，默认30s。
    - 通过该机制来保证不被故障节点影响。
  - 注：ISR中，应答的最小副本数量为1（min.insync.replicas），默认值也为1，即只有leader本身，也就是说，不修改该值，那么效果等同于 ack=1。

因此，确保数据完全可靠条件（理论上），ACK级别=-1，分区副本>=2，ISR中应答最小副本数量>=2。

总结：

- ack=0，生产环境基本不用。

- ack=1，一般用于传输普通日志，允许丢失个别数据。

- ack=-1，一般用于传输和钱（或类似重要级别的）相关的数据，对可靠性要求高。

```java
// 设置ack，默认 all或-1
properties.put(ProducerConfig.ACKS_CONFIG, "1");
// 重试次数，无应答时的重试次数，默认是int最大值
properties.put(ProducerConfig.RETRIES_CONFIG, 3);
```

### 数据重复

数据传递语义

- At Least Once，ack=-1 + 分区副本 >= 2 + ISR 应答的最小副本数量 >= 2。保证数据不丢失，但不能保证数据不重复。

- At Most Once，ack=0，保证数据不重复，但不能保证数据不丢失。

- Exactly Once，比如钱场景下，保证数据既不重复也不丢失。
  - Kafka0.11版本后，引入了幂等性和事务。幂等性就是不论 producer向 broker发送多少次重复数据，broker都只会持久化一条，避免消息重复。
    - 重复数据的判断标准：\<PID, partition, seqNumber> 相同主键的消息提交时，broker只会持久化一条。其中，
      - PID，每次kafka重启会分配一个新的。
      - partition表示分区号。
      - sequence number 是单调递增。
      - 所以，幂等性，只能保证在单分区会话内不重复。
  - 因此，exactly once = 幂等性 + at least once。

如何使用幂等性？

- 开启 enable.idempotence参数，默认为true，false关闭。

在kafka中开启事务的前提，必须开启幂等性。处理事务是由 Transaction Coordinator（事务协调器） 完成的。每个broker都有自己的事务协调器（不同broker之间的事务如何协调，可能和底层hashcode有关）。

```java
public static void setProducerTransaction(){
    Properties properties = new Properties();
    properties.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, "kafka01:9092,kafka02:9092");
    properties.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
    properties.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());

    // 指定事务ID，不然会报错
    properties.put(ProducerConfig.TRANSACTIONAL_ID_CONFIG, "testing_transaction_id_01");

    Producer<Object, Object> producer = new KafkaProducer<>(properties);

    // 初始化事务
    producer.initTransactions();
    // 开启事务
    producer.beginTransaction();

    try{
        for (int i = 0; i < 5; i++) {
            producer.send(new ProducerRecord<>("first", "test setting transaction" + i));
        }
        // 提交事务
        producer.commitTransaction();
    }catch (Exception e){
        // 出现异常，就终止事务
        producer.abortTransaction();
    }finally {
        producer.close();
    }
}
```

### 数据有序

单分区内，消息有序（有一定条件）；多分区下，分区之间数据无序。

### 数据乱序

InFlightRequests，默认每个broker最多缓存5个请求。正常情况下，每次请求集群都成功，那么消息就是顺序，如果前面的request第一次请求不成功，就会造成后面的请求完成在在前面的请求之前。就会造成乱序的情况（单分区）。

为了解决乱序，单分区下，分两种情况，

- 未开启幂等性，需要将 max.in.flight.requests.per.connnection 设置为1。
- 开启幂等性，需要将 max.in.flight.requests.per.connnection 设置 <= 5。

其底层是通过，开启幂等性后，kafka服务端会缓存producer发来的最近的5个 request的元数据，无论如何，都可以保证最近5个的request的数据是有序的。

## 4. Broker

## 5. 消费者

## 6. Eagle监控

## 7. Kraft模式

