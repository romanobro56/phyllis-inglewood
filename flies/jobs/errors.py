"""Typed adapter errors; ambiguous/transient failures never skip a job."""
BLOCKING = frozenset({'account_exists', 'already_applied', 'listing_closed',
                      'missing_required_information', 'unsupported_form'})
RETRYABLE = frozenset({'validation_error', 'network_timeout', 'browser_error'})


def validate_error(code):
    if code not in BLOCKING | RETRYABLE:
        raise ValueError(f'Unrecognized application error: {code}')
    return code
