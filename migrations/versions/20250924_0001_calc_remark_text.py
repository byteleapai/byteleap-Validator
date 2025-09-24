"""change calculation_remark to Text

Revision ID: 6a8c3a21b2f1
Revises: a4e37089d772
Create Date: 2025-09-24 00:01:00+00:00

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "6a8c3a21b2f1"
down_revision = "a4e37089d772"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("network_weights", schema=None) as batch_op:
        batch_op.alter_column("calculation_remark", type_=sa.Text())


def downgrade() -> None:
    with op.batch_alter_table("network_weights", schema=None) as batch_op:
        batch_op.alter_column("calculation_remark", type_=sa.String(length=256))
