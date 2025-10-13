from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import sqlalchemy.orm as orm
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/watchtower_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class BacktestRun(Base):
    __tablename__ = "backtest_runs"
    id = Column(Integer, primary_key=True, index=True)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    metrics = Column(JSON)
    equity_curve = Column(JSON)
    alloc_hist = Column(JSON)
    switches = Column(Integer)

class OHLCVData(Base):
    __tablename__ = "ohlcv_data"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

# Create tables if not exist
Base.metadata.create_all(bind=engine)