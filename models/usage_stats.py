"""
Quantonium OS - Usage Statistics Models

This module defines the SQLAlchemy models for tracking application usage.
"""

from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, Text, func
from auth.models import db

class AppUsage(db.Model):
    """Application usage tracking model"""
    __tablename__ = 'app_usage_stats'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(36), nullable=False)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    
    # Page/feature accessed
    endpoint = Column(String(255), nullable=False)
    feature = Column(String(50), nullable=True)  # encryption, decryption, quantum-grid, etc.
    
    # Access details
    timestamp = Column(DateTime, default=func.now())
    duration_ms = Column(Integer, nullable=True)  # Optional session duration
    
    # Additional metadata
    is_mobile = Column(Integer, default=0)  # 0=desktop, 1=mobile
    browser = Column(String(50), nullable=True)
    os = Column(String(50), nullable=True)
    
    @classmethod
    def log_access(cls, session_id, endpoint, feature=None, ip_address=None, 
                   user_agent=None, duration_ms=None, is_mobile=0, 
                   browser=None, os=None):
        """
        Log application access
        
        Args:
            session_id: Unique session identifier
            endpoint: The endpoint/route accessed
            feature: Optional feature name
            ip_address: Client IP address
            user_agent: Client user agent string
            duration_ms: Optional session duration in milliseconds
            is_mobile: Whether the client is mobile (0=desktop, 1=mobile)
            browser: Browser name if detected
            os: Operating system if detected
        """
        entry = cls(
            session_id=session_id,
            endpoint=endpoint,
            feature=feature,
            ip_address=ip_address,
            user_agent=user_agent,
            duration_ms=duration_ms,
            is_mobile=is_mobile,
            browser=browser,
            os=os
        )
        
        db.session.add(entry)
        db.session.commit()
        
        return entry
    
    @classmethod
    def get_usage_stats(cls):
        """
        Get usage statistics summary
        
        Returns dictionary with usage statistics
        """
        total_visits = db.session.query(func.count(cls.id)).scalar() or 0
        unique_visitors = db.session.query(func.count(func.distinct(cls.session_id))).scalar() or 0
        
        # Get most popular features
        feature_counts = db.session.query(
            cls.feature, 
            func.count(cls.id).label('count')
        ).filter(cls.feature.isnot(None)).group_by(cls.feature).order_by(
            func.count(cls.id).desc()
        ).limit(5).all()
        
        features = [{'feature': f[0], 'count': f[1]} for f in feature_counts]
        
        # Get most recent visits
        recent_visits = db.session.query(
            cls.endpoint,
            cls.feature,
            cls.timestamp,
            cls.ip_address
        ).order_by(cls.timestamp.desc()).limit(10).all()
        
        recent = [{'endpoint': v[0], 'feature': v[1], 'timestamp': v[2].isoformat(), 'ip': v[3]} 
                  for v in recent_visits]
        
        # Get visits by browser
        browser_counts = db.session.query(
            cls.browser, 
            func.count(cls.id).label('count')
        ).filter(cls.browser.isnot(None)).group_by(cls.browser).order_by(
            func.count(cls.id).desc()
        ).limit(5).all()
        
        browsers = [{'browser': b[0], 'count': b[1]} for b in browser_counts]
        
        # Calculate mobile percentage
        mobile_count = db.session.query(func.count(cls.id)).filter(cls.is_mobile == 1).scalar() or 0
        mobile_percentage = round((mobile_count / total_visits) * 100 if total_visits > 0 else 0, 2)
        
        return {
            'total_visits': total_visits,
            'unique_visitors': unique_visitors,
            'features': features,
            'recent_visits': recent,
            'browsers': browsers,
            'mobile_percentage': mobile_percentage
        }