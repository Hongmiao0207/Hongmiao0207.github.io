# kubernetes入门

 

## 1 核心组件

### 1.1 什么是 Node？

一个节点就是一个服务器或虚拟机。它会运行容器化应用程序。每个集群至少有一个工作节点。 一个集群通常运行多个节点，提供容错性和高可用性。

{{< mermaid >}}graph LR;
    A{Node}
    A -->|as| D[Server]
    A -->|as| E[VM]
{{< /mermaid >}}

在一个节点中可以运行一个或多个pod。

### 1.2 什么是Pod？

Pods are **the smallest deployable units** of computing that you can **create and manage** in Kubernetes. --- [Pod-官方文档](https://kubernetes.io/docs/concepts/workloads/pods/)

Pod 是Kubernetes的最小调度单元。它是一个或者多个容器的组合，由它来创建容器的运行环境。在这个环境中，容器会共享一些资源，比如存储、网络以及运行时的一些配置等。

假如，我们的系统包括一个应用程序和数据库，就可以把它们分别放到两个pod中。

注：一般建议一个pod中只运行一个容器，但pod本身并不是只能运行一个容器。目的是为了实现应用的解耦和扩展。高度耦合的容器除外，比如 边车模式（[Sidecar pattern](https://learn.microsoft.com/en-us/azure/architecture/patterns/sidecar)）。

再回到我们的假设中，应用程序要访问数据库的话就需要知道数据库的IP地址，而这个IP地址会在pod创建的时候自动分配，它是一个集群内部的IP地址，在node中pods可以通过IP地址来完成通信。请注意，pod本身是一个不稳定实体，这意味它更容易被创建和销毁。当某个pod故障时，kubernetes会自动将该pod销毁，并创建个新pod来替代它。此时它的IP地址发生变化，其他服务该如何访问它？通过一个叫 Service的资源对象来实现。

### 1.3 什么是Service？

In Kubernetes, a Service is a method for exposing a network application that is running as one or more Pods in your cluster. --- [Service-官方文档](https://kubernetes.io/docs/concepts/services-networking/service/)

Service可以将一组Pod可在网络上访问，客服端就能与之交互。比如，我们可以将之前的应用程序和数据库的Pods分别封装成两个Service，这样应用程序就可以用过Service提供的IP地址来访问数据库。它通过客户端只需要请求Service，通过Service转发请求到健康的数据库Pod上。这样就解决的PodIP地址不稳定的问题。

服务也分为内部服务和外部服务，内部服务是不想或者没有必要暴露给外部的服务，比如数据库、缓存、消息队列等。外部服务包括，API接口、前端界面等。

外都服务有几种常用类型，比如 NodePort，它会在节点上开放一个端口，将这个端口映射到 Service的IP地址和端口上，这样就可以用过节点的IP地址和端口来访问Service。但是在生产环境中，我们通常使用域名来访问服务，这个时候就会用到另一个资源对象 Ingress。

### 1.4 什么是 Ingress？

An API object that manages external access to the services in a cluster, typically HTTP. --- [Ingress](https://kubernetes.io/docs/concepts/services-networking/ingress/)

Ingress 是用来管理集群外部访问集群内部服务的入口和方式，比如HTTP、HTTPS路由。可以通过Ingress来配置不同的转发规则，从而根据不同的规则来访问集群内部不同的Service以及Service对应的后端Pods，或者通过Ingress来配置域名。另外它还提供了负载均衡（load balancing）、SSL终结（SSL termination）和基于名称的虚拟托管（name-based virtual hosting）等功能。

值得一提的是，Ingress目前已经停止更新，新的功能都集成到了 [Gateway API](https://kubernetes.io/docs/concepts/services-networking/gateway/) 中。

### 1.5 什么是 ConfigMaps？

虽然我们的应用程序和数据库已经可以在Kubernetes中运行并对外提供服务，但是这时还有一个问题，就是二者的耦合问题。比如，应用程序访问数据库的通常做法是将数据库地址、用户名和密码等连接信息写到配置文件或环境变量中。这样的弊端就是应用程序和这些配置信息耦合了，一旦数据库地址发生变化，我们就要重新编译应用程序再部署到集群中。而 ConfigMaps 可以将配置信息封装并提供给应用程序读取和使用，也就是说，它会将配置信息和应用程序的镜像内容解耦。

A ConfigMap is an API object used to store non-confidential data in key-value pairs. Pods can consume ConfigMaps as environment variables, command-line arguments, or as configuration files in a volume. --- [ConfigMaps-官方文档](https://kubernetes.io/docs/concepts/configuration/configmap/)

ConfigMap 保证了应用程序的可移植性，我们只需要修改ConfigMap对象中的配置信息，然后重新加载Pod就可以了，避免了重新编译和部署应用程序。

注意，ConfigMap 中的配置信息都是明文的，它并不提供保密或者加密功能。如果想要存储的数据的敏感的，请用 [Secret](https://kubernetes.io/docs/concepts/configuration/secret/) （仅做了一层base64编码而非加密）或者第三方工具来加密。

### 1.6 什么是 Volumes？

当容器重启或被销毁的时候，容器中的数据也会跟着消失，这对于数据库这种需要数据持久化的应用程序来说是不可接受的。而 Volumes 就是用来支持消息持久化的。它支持将资源持久化挂载到集群本地磁盘或集群外部远程存储上。

On-disk files in a container are ephemeral, which presents some problems for non-trivial applications when running in containers. One problem occurs when a container crashes or is stopped. Container state is not saved so all of the files that were created or modified during the lifetime of the container are lost. During a crash, kubelet restarts the container with a clean state. Another problem occurs when multiple containers are running in a `Pod` and need to share files. It can be challenging to setup and access a shared filesystem across all of the containers. The Kubernetes volume abstraction solves both of these problems. --- [Volumes-官方文档](https://kubernetes.io/docs/concepts/storage/volumes/)

### 1.7 什么是 Deployments？

Deployments保证了应用程序的高可用性，比如，应用程序的节点发生了故障或者需要对节点进行升级或更新维护需要暂停服务。Deployment的解决方案就是创建多个节点，当一个节点无法请求的时候，Service就会转发到另一个节点来继续提供服务。

A Deployment provides declarative updates for Pods and ReplicaSets. --- [Deployments-官方文档](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)

Deployments可以定义和管理应用程序的副本数量以及应用程序的更新策略，旨在简化应用程序的部署和更新操作。

之前我们有提到 Pods 可以理解为在容器上加了一层抽象，这样就可以将一个或多个容器组合在一起。而 Deployments 就可以理解为在 Pods 上面再加一层抽象，这样就可以将一个或多个 Pods 组合在一起。并且它还具有副本控制、滚动更新、自动扩缩容等高级特性和功能。

- 副本控制：定义和管理应用程序的副本数量。比如定义一个应用程序的副本数量为3，那么有副本发生故障时，Deployments 会自动创建一个新的副本来替代它，始终保持有3个副本在集群中运行。
- 滚动更新：定义和管理应用程序的更新策略，更轻松地升级应用程序的版本，保证程序的平滑升级。

数据库也会有类似的问题，比如，数据库的节点发生了故障、升级和更新维护时，也需要停止服务。所以数据库也会采取类似的多副本的方式来保证它的高可用性。但和应用程序不同的是，一般不会使用  Deployments 来实现数据库的多副本。因为数据库的多副本之间是有状态的。简单来说就是每个副本都有自己的状态（也就是数据），这些数据都需要持久化存储，同时也要保证多个副本之间的数据是一致的，这就需要一些额外的机制，比如把数据写入到一个共享存储中或者把数据同步到其他副本中。对于这种有状态的应用程序管理，需要用到Kubernetes中的 StatefulSet。

### 1.8 什么是 StatefulSet？

StatefulSet is the workload API object used to manage stateful applications.Manages the deployment and scaling of a set of Pods, and provides guarantees about the ordering and uniqueness of these Pods. --- [StatefulSet-官方文档](https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/)

和 Deployment 类似， StatefulSet 管理也提供了副本控制、滚动更新、自动扩缩容等高级特性和功能。但和 Deployment 不同的是， StatefulSet 为它们的每个 Pod 维护了一个稳定的、唯一的网络标识符。这些 Pod 是基于相同的规约来创建的， 但是不能相互替换：无论怎么调度，每个 Pod 都有一个永久不变的 ID。它还提供了稳定的、持久化的存储。因此像数据库、缓存、消息队列等有状态的应用以及保留了会话状态的应用程序一般都需要使用 StatefulSet来部署。

其弊端就是部署的过程相对复杂和繁琐。并且不是所有的有状态的应用程序都是用于 StatefulSet类部署。一种更加通用和简单的方式是把数据库这种有状态的应用程序从Kubernetes集群中剥离出来，在集群外单独部署，从而避免不必要的麻烦和问题，并且可以简化集群的架构和管理。

## 2 Kubernetes架构

Kubernetes是一个典型的 [Master-Worker架构](https://zhuanlan.zhihu.com/p/386360903)，Master-Node负责管理整个集群，Worker-Node负责运行应用程序和服务。

Kubernetes runs your workload by placing containers into Pods to run on Nodes. A node may be a virtual or physical machine, depending on the cluster. Each node is managed by the control plane and contains the services necessary to run Pods. --- [Kubernetes-官方节点](https://kubernetes.io/docs/concepts/architecture/nodes/)
