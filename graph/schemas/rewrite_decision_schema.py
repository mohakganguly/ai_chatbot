from pydantic import BaseModel


class RewriteDecision(BaseModel):

    rewrite: bool