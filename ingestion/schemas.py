"""Pydantic schemas describing the structured data we extract from documents.

Each field's `description` is fed to the LLM via `with_structured_output`: it is
the instruction the model reads to decide what to pull from the raw text. The
source documents are Romanian, but field names and descriptions stay English —
the LLM is multilingual and maps the Romanian text onto these fields.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Product(BaseModel):
    name: str = Field(description="Name of the product or service.")
    quantity: float | None = Field(
        default=None, description="Quantity ordered for this product."
    )
    unit_price: float | None = Field(
        default=None, description="Unit price before VAT for this product."
    )
    line_total: float | None = Field(
        default=None, description="Line total (quantity * unit price), before VAT."
    )


class Invoice(BaseModel):
    number: str = Field(description="Invoice number, e.g. 'FV-2024-001'.")
    date: str | None = Field(
        default=None, description="Issue date exactly as written on the invoice."
    )
    client: str | None = Field(
        default=None, description="Buyer / client company name."
    )
    supplier: str | None = Field(
        default=None, description="Supplier / issuer company name."
    )
    total: float | None = Field(
        default=None, description="Total amount due, including VAT."
    )
    products: list[Product] = Field(
        default_factory=list,
        description="Products or services listed on the invoice.",
    )


class Contract(BaseModel):
    number: str = Field(description="Contract number, e.g. 'CS-2024-015'.")
    date_concluded: str | None = Field(
        default=None, description="Date the contract was concluded, as written."
    )
    provider: str | None = Field(
        default=None, description="Service provider company name."
    )
    beneficiary: str | None = Field(
        default=None, description="Beneficiary / client company name."
    )
    value: float | None = Field(
        default=None, description="Total contract value (numeric amount)."
    )
    duration_months: int | None = Field(
        default=None, description="Contract duration in months, if stated."
    )
    provider_obligations: list[str] = Field(
        default_factory=list,
        description="The provider's obligations / services to be delivered.",
    )
