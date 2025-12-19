# ByteLeap Validator - Bittensor SN128 Compute Network

ByteLeap Validator is the network coordination node for Bittensor SN128, managing challenge validation, weight calculation, and network scoring for the distributed compute resource platform.

## Architecture Overview

**Validator Responsibilities:**
- **Challenge Validation**: Two-phase verification protocol for computational integrity
- **Weight Management**: Network-wide scoring and weight updates
- **Resource & Lease Tracking**: PostgreSQL-based miner and worker performance monitoring
- **Secure Communication**: Session-based encryption with miners
- **VM Gateway Enrollment Brokering**: Retrieves, caches, and serves VM gateway enroll tokens from MeshHub to miners

## Scoring System

The validator manages a dual-factor scoring system for network participants:

### Score Components (Weighted)
- **Lease Revenue**: Active compute rentals generate the primary score
- **Challenge Performance**: Computational benchmarks for idle workers
- **Availability Multiplier**: Scales challenge output using the configured uptime window

### How Scoring Works

**Lease Revenue**
- Workers with active compute rentals earn lease scores
- Idle workers score zero on this component
- Integrated with compute marketplace APIs

**Challenge Performance**
- CPU/GPU matrix multiplication benchmarks
- Two-phase verification prevents cheating:
  - Phase 1: Workers commit to results (merkle root)
  - Phase 2: Validators verify through random sampling
- Scoring uses participation baseline + absolute performance scoring
- Rewards consistent participation over peak performance

**Worker Management**
- Weight aggregation considers only a capped worker set per miner
- Challenges target only unleased workers
- Final score sums all eligible worker performance while respecting the per-miner cap

## Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL 12+
- Bittensor wallet with registered hotkey
- Sufficient TAO stake for network participation

### Installation

```bash
# Setup environment
python3 -m venv venv
source ./venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup PostgreSQL database, skip this if you use sqlite (default config)
# (cp scripts/setup_database.sh /tmp; cd /tmp; sudo -u postgres /tmp/setup_database.sh setup)
```

### Configuration

Configure your validator in `config/validator_config.yaml`:
- Network settings (netuid, wallet paths)
- Database connection parameters
- Challenge verification settings
- Weight update intervals
- MeshHub connectivity (`meshhub.ws_url`, `meshhub.access_token`, capabilities)
- Security controls such as rate limiting and proof cache size

### Running the Validator

**Start Validator**:
```bash
python scripts/run_validator.py --config config/validator_config.yaml
```

**Database Management**:
```bash
# Apply database migrations
./scripts/db_migrate.py upgrade

# Check database connection
./scripts/db_migrate.py check
```

## Technical Architecture

```
┌────────────────────────┐                       ┌───────────────────┐                   ┌────────────────────────┐
│       Validator        │                       │       Miner       │                   │       Worker(s)        │
│      (Bittensor)       │       Encrypted       │    (Bittensor)    │                   │                        │
│                        │ ←── Communication ─── │                   │ ←── WebSocket ──→ │ • System Monitoring    │
│ • Challenge Creation   │    (via bittensor)    │ • Worker Mgmt.    │      (1 : N)      │ • Challenge Execution  │
│ • Score Validation     │                       │ • Resource Agg.   │                   │ • VMGW Session         │
│ • Weight Calculation   │                       │ • Task Routing    │                   │ • Libvirt Mgmt.        │
└────────────────────────┘                       └───────────────────┘                   └────────────────────────┘
```

### Core Components

**Validator Core** (`neurons/validator/`)
- `core/validator.py` - Main validator orchestration, MeshHub client wiring, Axon handlers
- `services/validation.py` - Challenge validation engine with GPU allowlist enforcement
- `services/weight_manager.py` - Weight calculation, metagraph gating, subtensor guard integration
- `services/async_challenge_verifier.py` - Asynchronous proof verification
- `services/communication.py` - Encrypted synapse routing with per-hotkey rate limiting
- `services/data_cleanup.py` - Nightly retention and availability-safe pruning

**Database Models** (`neurons/validator/models/`)
- `MinerInfo` - Miner registration and weight tracking
- `WorkerInfo` - Individual worker performance metrics (lease sync, availability)
- `ComputeChallenge` - Challenge tracking with verification state
- `NetworkWeight` - Historical weight calculations
- `GPUInventory` - GPU fingerprinting and allowlist checks
- `HeartbeatRecord` - Worker heartbeat history retained per policy

## Operational Services

- **MeshHub WebSocket**: Streams heartbeats, resource reports, lease updates, and VM gateway token responses over an encrypted session with cached retries.
- **Proof Cache**: `validation.proof_queue_max_size` bounds in-memory proofs; evictions trigger cleanup to avoid dangling verifications.
- **Rate Limiter**: `security.rate_limit.*` defines the sliding window used to throttle abusive miners before synapse handlers run expensive work.
- **Data Cleanup**: Retention windows ensure historical tables stay within size targets while preserving the availability horizon used for scoring.

## Development

### Database Operations

```bash
# PostgreSQL Setup (RHEL/CentOS)
yum install postgresql-server postgresql-contrib
/usr/bin/postgresql-setup --initdb
systemctl restart postgresql

# Configure access
vi /var/lib/pgsql/data/pg_hba.conf
# Add: host all all 127.0.0.1/32 md5
systemctl restart postgresql
```

## License

MIT License - see the [LICENSE](LICENSE) file for details.
