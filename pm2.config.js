module.exports = {
  apps: [
    {
      name: 'subnet-validator',
      script: './scripts/run_validator.py',
      interpreter: 'python3',
      args: '--config config/validator_config.yaml',
      cwd: process.cwd(),
      
      // Process management
      instances: 1,
      exec_mode: 'fork',
      
      // Auto restart configuration
      autorestart: true,
      max_restarts: 10,
      min_uptime: '10s',
      max_memory_restart: '8G',
      
      // Logging configuration
      log_file: 'logs/pm2/validator-combined.log',
      out_file: 'logs/pm2/validator-out.log',
      error_file: 'logs/pm2/validator-error.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true,
      
      // Environment variables
      env: {
        NODE_ENV: 'production',
        PYTHONPATH: process.cwd(),
        PYTHONUNBUFFERED: '1'
      },
      
      // Development environment
      env_development: {
        NODE_ENV: 'development',
        PYTHONPATH: process.cwd(),
        PYTHONUNBUFFERED: '1'
      },
      
      // Graceful shutdown
      kill_timeout: 30000,
      listen_timeout: 10000,
      
      // Health monitoring
      health_check_grace_period: 3000,
      
      // Watch and ignore patterns (optional - for development)
      watch: false,
      ignore_watch: [
        'node_modules',
        'logs',
        '.git',
        '*.log',
        'migrations/versions'
      ],
      
      // Custom startup delay
      wait_ready: true,
      
      // Time zone
      time: true
    }
  ]
}
