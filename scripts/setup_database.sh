#!/bin/bash
# Database setup script
# Used to set up PostgreSQL database and user

set -e

# Configuration
DB_NAME="neurons"
DB_USER="neurons"
DB_PASSWORD="PrygvMv3U5KjBEKtye7S"
DB_HOST="localhost"
DB_PORT="5432"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if PostgreSQL is installed
check_postgresql() {
    if ! command -v psql &> /dev/null; then
        log_error "PostgreSQL not installed"
        echo "Please install PostgreSQL:"
        echo "  Ubuntu/Debian: sudo apt-get install postgresql postgresql-contrib"
        echo "  CentOS/RHEL: sudo yum install postgresql-server postgresql-contrib"
        echo "  macOS: brew install postgresql"
        exit 1
    fi
    
    log_info "PostgreSQL installed"
}

# Check PostgreSQL service status
check_postgresql_service() {
    if systemctl is-active --quiet postgresql 2>/dev/null; then
        log_info "PostgreSQL service is running"
        return 0
    elif brew services list 2>/dev/null | grep postgresql | grep started &>/dev/null; then
        log_info "PostgreSQL service is running (Homebrew)"
        return 0
    else
        log_warning "PostgreSQL service is not running"
        return 1
    fi
}

# Start PostgreSQL service
start_postgresql() {
    log_info "Attempting to start PostgreSQL service..."
    
    if command -v systemctl &> /dev/null; then
        sudo systemctl start postgresql
    elif command -v brew &> /dev/null; then
        brew services start postgresql
    else
        log_error "Cannot start PostgreSQL service, please start manually"
        exit 1
    fi
    
    sleep 3
    
    if check_postgresql_service; then
        log_info "PostgreSQL service started successfully"
    else
        log_error "PostgreSQL service failed to start"
        exit 1
    fi
}

# Create database user
create_user() {
    log_info "Creating database user: $DB_USER"
    
    # Check if user already exists
    if psql -d postgres -tAc "SELECT 1 FROM pg_roles WHERE rolname='$DB_USER'" | grep -q 1; then
        log_warning "User '$DB_USER' already exists"
        return 0
    fi
    
    psql -d postgres -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';"
    psql -d postgres -c "ALTER USER $DB_USER CREATEDB;"
    
    log_info "User '$DB_USER' created successfully"
}

# Create database
create_database() {
    log_info "Creating database: $DB_NAME"
    
    # Check if database already exists
    if psql -lqt | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
        log_warning "Database '$DB_NAME' already exists"
        return 0
    fi
    
    createdb -O "$DB_USER" "$DB_NAME"
    
    log_info "Database '$DB_NAME' created successfully"
}

# Set permissions
setup_permissions() {
    log_info "Setting up database permissions"
    
    psql -d "$DB_NAME" -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"
    psql -d "$DB_NAME" -c "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO $DB_USER;"
    psql -d "$DB_NAME" -c "GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO $DB_USER;"
    psql -d "$DB_NAME" -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO $DB_USER;"
    psql -d "$DB_NAME" -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO $DB_USER;"
    
    log_info "Permissions setup completed"
}

# Test connection
test_connection() {
    log_info "Testing database connection"
    
    export PGPASSWORD="$DB_PASSWORD"
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT version();" &>/dev/null; then
        log_info "Database connection test successful"
        unset PGPASSWORD
        return 0
    else
        log_error "Database connection test failed"
        unset PGPASSWORD
        return 1
    fi
}

# Create configuration file template
create_config_template() {
    log_info "Creating database configuration template"
    
    cat > /tmp/database_config.yaml << EOF
# Database configuration
database:
  url: postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME
  pool_size: 10
  max_overflow: 20

# Connection parameters
connection:
  host: $DB_HOST
  port: $DB_PORT
  username: $DB_USER
  password: $DB_PASSWORD
  database: $DB_NAME
EOF
    
    log_info "Configuration template generated: /tmp/database_config.yaml"
}

# Show connection information
show_connection_info() {
    echo ""
    echo "=== Database Connection Information ==="
    echo "Host: $DB_HOST"
    echo "Port: $DB_PORT"
    echo "Database: $DB_NAME"
    echo "Username: $DB_USER"
    echo "Password: $DB_PASSWORD"
    echo ""
    echo "Connection URL:"
    echo "postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME"
    echo ""
}

# Clean up database
cleanup_database() {
    log_warning "Cleaning up database and user"
    
    read -p "Are you sure you want to delete database '$DB_NAME' and user '$DB_USER'? [y/N]: " confirm
    if [[ $confirm != [yY] ]]; then
        log_info "Cleanup operation cancelled"
        return 0
    fi
    
    # Delete database
    if psql -lqt | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
        dropdb "$DB_NAME"
        log_info "Database '$DB_NAME' deleted"
    fi
    
    # Delete user
    if psql -d postgres -tAc "SELECT 1 FROM pg_roles WHERE rolname='$DB_USER'" | grep -q 1; then
        psql -d postgres -c "DROP USER $DB_USER;"
        log_info "User '$DB_USER' deleted"
    fi
}

# Show help
show_help() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  setup     - Complete database setup (default)"
    echo "  test      - Test database connection"
    echo "  info      - Show connection information"
    echo "  cleanup   - Clean up database and user"
    echo "  help      - Show this help"
    echo ""
}

# Complete setup process
setup_complete() {
    log_info "Starting PostgreSQL database setup"
    
    # Check PostgreSQL
    check_postgresql
    
    # Check and start service
    if ! check_postgresql_service; then
        start_postgresql
    fi
    
    # Create user and database
    create_user
    create_database
    setup_permissions
    
    # Test connection
    if test_connection; then
        create_config_template
        show_connection_info
        log_info "Database setup completed!"
    else
        log_error "Database setup failed"
        exit 1
    fi
}

# Main function
main() {
    case "${1:-setup}" in
        setup)
            setup_complete
            ;;
        test)
            test_connection
            ;;
        info)
            show_connection_info
            ;;
        cleanup)
            cleanup_database
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"