# ByteLeap Validator - Bittensor SN128 Compute Network

ByteLeap Validator is the network coordination node for Bittensor SN128, managing challenge validation, weight calculation, and network scoring for the distributed compute resource platform.

## Architecture Overview

**Validator Responsibilities:**
- **Challenge Validation**: Two-phase verification protocol for computational integrity
- **Weight Management**: Network-wide scoring and weight updates
- **Resource Tracking**: PostgreSQL-based miner and worker performance monitoring
- **Secure Communication**: Session-based encryption with miners

## Scoring System

The validator manages a dual-factor scoring system for network participants:

### Score Components (Weighted)
- **Lease Revenue** (70%): Active compute rentals generate the primary score
- **Challenge Performance** (30%): Computational benchmarks for idle workers
- **Availability Multiplier**: Based on 169-hour online presence

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
- Maximum 100 workers per miner
- Challenges target only unleased workers
- Final score sums all worker performance (capped at 100)

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
┌────────────────────────┐                       ┌───────────────────┐
│      Validator         │       Encrypted       │       Miner       │
│     (Bittensor)        │ ←── Communication ─── │    (Bittensor)    │
│                        │    (via bittensor)    │                   │
│ • Challenge Creation   │                       │ • Worker Mgmt.    │
│ • Score Validation     │                       │ • Resource Agg.   │
│ • Weight Calculation   │                       │ • Task Routing    │
└────────────────────────┘                       └───────────────────┘
                                                          ↑
                                                          │ WebSocket
                                                          │
                                               ┌───────────────────────┐
                                               │      Worker(s)        │
                                               │                       │
                                               │ • System Monitoring   │
                                               │ • Challenge Execution │
                                               │ • Compute Tasks       │
                                               └───────────────────────┘
```

### Core Components

**Validator Core** (`neurons/validator/`)
- `core/validator.py` - Main validator orchestration
- `services/validation.py` - Challenge validation engine
- `services/weight_manager.py` - Weight calculation and network updates
- `services/async_challenge_verifier.py` - Asynchronous proof verification

**Database Models** (`neurons/validator/models/`)
- `MinerInfo` - Miner registration and weight tracking
- `WorkerInfo` - Individual worker performance metrics
- `ComputeChallenge` - Challenge tracking with verification state
- `NetworkWeight` - Historical weight calculations

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
