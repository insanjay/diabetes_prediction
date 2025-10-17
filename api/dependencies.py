from database.database_setup import SessionLocal

def get_db():
    """
    FastAPI dependency to get a database session for each API request.
    This function uses a generator (yield) to provide the session.
    The 'finally' block ensures the database session is always closed
    after the request is finished, which is crucial for preventing
    database connection leaks.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

