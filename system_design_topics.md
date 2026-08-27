# System Design for Interviews — Fresher/Entry-Level Prep
### Priority-Ranked Topic List with Sub-Topics

---

## Tier 1: Must-Know (80% of interview weight — learn these first)

### 1. Scalability Basics
- Vertical scaling vs horizontal scaling
- Stateless vs stateful services
- Scaling reads vs scaling writes
- Bottleneck identification

### 2. Load Balancing
- L4 (transport layer) vs L7 (application layer) load balancing
- Algorithms: round robin, weighted round robin, least connections, consistent hashing, IP hash
- Health checks
- Sticky sessions

### 3. Caching
- Cache hit vs cache miss
- Eviction policies: LRU, LFU, FIFO
- Write strategies: write-through, write-back, write-around
- Client-side vs server-side vs CDN caching
- Redis vs Memcached basics
- Cache invalidation strategies (TTL, event-based)

### 4. Database Fundamentals
- SQL vs NoSQL — when to use which
- Indexing (B-Tree, hash index) and trade-offs
- Normalization vs denormalization
- Replication basics: master-slave, master-master, failover
- Primary key/foreign key design basics
- Database types: relational, key-value, document, columnar, graph

### 5. CAP Theorem
- Consistency, Availability, Partition tolerance — definitions
- Why you can only pick 2 of 3 during a partition
- CP vs AP systems with examples (e.g., MongoDB vs Cassandra)
- Real-world implications for design choices

### 6. API Design Basics
- REST principles (resources, verbs, statelessness)
- Rate limiting: token bucket, leaky bucket, sliding window
- Pagination strategies (offset-based, cursor-based)
- Versioning strategies
- Status codes and idempotency

### 7. Back-of-Envelope Estimation
- Estimating QPS (queries per second)
- Storage estimation (per user/per record × scale)
- Bandwidth estimation
- Read-heavy vs write-heavy system identification
- Converting requirements into numbers before designing

### 8. Core HLD Practice Problems
- URL Shortener (TinyURL)
- Rate Limiter
- Pastebin
- (These three alone cover the majority of fresher-level rounds)

---

## Tier 2: High-Value (learn next — differentiates good candidates)

### 9. Sharding / Partitioning
- Range-based partitioning
- Hash-based partitioning
- Geo-based partitioning
- Hot partition / hotspot problem

### 10. Message Queues & Async Processing
- Why asynchronous communication is needed
- Pub-sub vs point-to-point messaging
- Kafka basics (topics, partitions, consumers)
- RabbitMQ / SQS basics
- Idempotency in message processing
- Event-driven architecture basics

### 11. Consistency Models
- Strong consistency
- Eventual consistency
- Causal consistency
- Trade-offs vs latency/availability

### 12. CDN (Content Delivery Network)
- What problem it solves (latency, load reduction)
- Static vs dynamic content delivery
- Edge servers / points of presence
- When to use a CDN

### 13. Database Replication (Deep Dive)
- Synchronous vs asynchronous replication
- Leader-follower failover mechanics
- Read replicas for scaling reads
- Replication lag issues

### 14. Authentication / Authorization
- Sessions vs tokens
- JWT structure and usage
- OAuth 2.0 basics
- API keys vs user auth

### 15. Proxies
- Reverse proxy vs forward proxy
- Use cases: SSL termination, load distribution, caching, anonymity

### 16. Additional HLD Practice Problems
- Chat application (WhatsApp-lite)
- Notification system
- Key-value store design
- News feed system (basic version)

---

## Tier 3: Good-to-Have (adds polish, rarely blocking)

### 17. Microservices vs Monolith
- Trade-offs (deployment, scaling, complexity)
- Synchronous vs asynchronous service communication
- Database-per-service pattern (conceptual)

### 18. Reliability Patterns
- Retry mechanisms with exponential backoff
- Circuit breaker pattern
- Bulkhead pattern
- Timeout handling

### 19. Service Discovery
- Client-side vs server-side discovery
- Service registry concept

### 20. Storage Systems
- File storage vs block storage vs object storage
- Blob storage basics (S3-like systems)

### 21. Distributed Transactions
- Two-phase commit (2PC) — conceptual understanding only
- Why distributed transactions are hard

### 22. Monitoring & Logging Awareness
- Health check endpoints
- Basic metrics (latency, error rate, throughput)
- Centralized logging concept
- Alerting basics

---

## Tier 4: LLD — Parallel Track (often bundled into the same interview slot)

### 23. OOP + SOLID Principles
- Encapsulation, inheritance, polymorphism, abstraction
- Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion

### 24. Design Patterns
- Singleton
- Factory
- Observer
- Strategy
- Builder

### 25. UML Basics
- Class diagrams
- Relationships: association, aggregation, composition, inheritance

### 26. Common LLD Practice Problems
- Parking lot system
- Elevator system
- Library management system
- Tic-tac-toe / board game design
- Vending machine

---

## How Freshers Are Usually Evaluated
- Clarifying requirements before diving into design
- Doing basic estimation math out loud
- Drawing a simple architecture (client → load balancer → servers → cache/DB)
- Discussing trade-offs (not expected to know deep distributed systems internals)
- Communicating thought process clearly — this matters more than a "perfect" answer

---

## The 80/20 Rule (If You Only Have 2 Weeks)

**Priority order:**
1. Scaling (vertical/horizontal, stateless/stateful)
2. Caching (eviction policies, write strategies)
3. Load balancing (algorithms, L4 vs L7)
4. Databases (SQL vs NoSQL, indexing, replication)
5. CAP theorem
6. Estimation math (QPS, storage, bandwidth)
7. Practice: Rate Limiter design (full answer)
8. Practice: URL Shortener design (full answer)

This combination covers the core concepts *and* gives you two fully-practiced example answers — usually enough to clear a fresher-level system design round.

---

## Suggested Study Sequence
Foundations → Networking Basics → Scaling → Databases → Caching → Load Balancing → API Design → Async/Message Queues → Practice HLD Problems → LLD Basics (in parallel throughout)
