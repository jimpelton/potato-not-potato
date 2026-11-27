from pydantic import BaseModel


class ReceiptItem(BaseModel):
    name: str
    quantity: float | None
    unit_price: float | None
    price: float


class Receipt(BaseModel):
    items: list[ReceiptItem]
    subtotal: float
    tax: float
    total: float
