from consts import ENV_LOCAL, ENV_NEWTON

def get_env_constant(e):
    if e == 'local':
        return ENV_LOCAL
    elif e== 'newton':
        return ENV_NEWTON
    else:
        ValueError('Invalid env:', e, '; Please pass a valid env.')