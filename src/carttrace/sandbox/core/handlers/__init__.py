"""One handler per command type. A handler changes the working copy through the context and either returns None, or
returns an error whose side effects must stay (a declined payment), or raises one whose side effects must not."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from carttrace.sandbox.core import commands as C
from carttrace.sandbox.core.context import Ctx
from carttrace.sandbox.core.errors import CoreError
from carttrace.sandbox.core.handlers import carts, checkouts, orders, payments, rails, system

Handler = Callable[[Ctx, Any], CoreError | None]

HANDLERS: Mapping[type[C.Command], Handler] = {
    C.OpenSession: system.open_session,
    C.RevokeToken: system.revoke_token,
    C.Read: system.read,
    C.CreateCart: carts.create_cart,
    C.UpdateCart: carts.update_cart,
    C.CancelCart: carts.cancel_cart,
    C.HandoffCart: carts.handoff_cart,
    C.CreateCheckout: checkouts.create_checkout,
    C.UpdateCheckout: checkouts.update_checkout,
    C.CancelCheckout: checkouts.cancel_checkout,
    C.CompleteCheckout: payments.complete_checkout,
    C.ConfirmPaymentAction: payments.confirm_payment_action,
    C.FulfillOrder: orders.fulfill_order,
    C.CancelOrder: orders.cancel_order,
    C.RefundOrder: orders.refund_order,
    C.EditOrder: orders.edit_order,
    C.RequestReturn: orders.request_return,
    C.ReceiveReturn: orders.receive_return,
    C.VaultCredential: rails.vault_credential,
    C.MintShopToken: rails.mint_shop_token,
    C.RequestPayment: rails.request_payment,
    C.RecordMandate: rails.record_mandate,
    C.SetStock: system.set_stock,
    C.SetPrice: system.set_price,
}
