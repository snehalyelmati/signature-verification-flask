from datetime import datetime
from signature_verification import db, login_manager
from flask_login import UserMixin

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(25), unique=True, nullable=False)
    email = db.Column(db.String(25), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    address = db.Column(db.Text, nullable=False)
    represents_company = db.Column(db.Integer, nullable=False, default=0)
    company_name = db.Column(db.String(25), nullable=False, default='None Specified')
    no_of_employees = db.Column(db.Integer, nullable=False, default=1)
    registered_on = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    history = db.relationship('History', backref='executed_by', lazy=True)
    notifications = db.relationship('Notification', backref='notifications', lazy=True)

    def __repr__(self):
        return f"User(Name:'{self.name}', Email:'{self.email}', Represents Company:'{self.represents_company}', Company:'{self.company_name}', Users:'{self.no_of_employees}')"

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    forgery_result = db.Column(db.Integer, nullable=False, default=0)
    user_result = db.Column(db.Integer, nullable=False, default=0)
    executed_on = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f"History(Result:'{self.result}', Executed On:'{self.executed_on}', Executed By:'{self.user_id}')"

class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    from_user = db.Column(db.String(25), nullable=False)
    to_user = db.Column(db.String(25), db.ForeignKey('user.email'), nullable=False)
    message = db.Column(db.String(100), nullable=False)
    at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f"Notification(From:'{self.from_user}', To:'{self.to_user}', Message:'{self.message}', at:'{self.at}')"