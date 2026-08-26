# UrbanAgent Admin Authentication Setup

## Security model

UrbanAgent Admin access is granted only when all of these checks pass:

1. The browser sends a Firebase ID token in the `Authorization: Bearer` header.
2. Firebase Admin SDK verifies the token.
3. The verified token contains the custom claim `admin: true`.
4. The request passes the centralized `/api/admin/*` authentication,
   authorization, and rate-limit middleware.

Browser state, a route, an email address, a Firestore profile role, or a request
body cannot grant Admin access.

## Configure the operator environment

Run the claim tool only from a trusted operator machine. Configure Firebase
Admin using the same supported environment mechanism as the backend:

- `FIREBASE_SERVICE_ACCOUNT_JSON`, or
- `FIREBASE_SERVICE_ACCOUNT_BASE64`, or
- Application Default Credentials through `GOOGLE_APPLICATION_CREDENTIALS`.

Never put a service-account credential in frontend configuration, command
history, screenshots, logs, or Git.

## Grant Admin access

Resolve the target Firebase Authentication UID in the Firebase console or an
approved operator workflow, then run:

```powershell
node scripts/set-admin-claim.js --uid <FIREBASE_UID> --admin=true
```

The tool preserves unrelated custom claims and sets only `admin: true`. It does
not accept a token or credential on the command line and does not expose one in
its output.

## Refresh the client token

Custom claims do not change an already-issued Firebase ID token. After the claim
is set, the user must sign out and sign in again. The dedicated Admin sign-in
flow also forces one token refresh after a successful sign-in.

An explicit client role refresh may call `getIdToken(true)`. Do not force a token
refresh on every render.

## Verify access

After refreshing the ID token, request:

```text
GET /api/admin/me
Authorization: Bearer <FIREBASE_ID_TOKEN>
```

Expected outcomes:

- `200`: token is valid and its verified `admin` claim is `true`.
- `401`: authentication is missing, malformed, expired, or invalid.
- `403`: authentication is valid, but the verified `admin` claim is not `true`.

The response returns only the verified UID, email, and Admin state. It never
returns all token claims.

## Revoke Admin access

Run:

```powershell
node scripts/set-admin-claim.js --uid <FIREBASE_UID> --admin=false
```

The user must then sign out and sign in again or refresh the ID token. For urgent
revocation, the operator should also revoke the user's refresh tokens through an
approved Firebase operator workflow.

## Read-only boundary

The current Admin foundation is read-only. It supports identity, capabilities,
Firebase Auth user listing, canonical POI summary, and safe system health.
Unknown Admin routes return `404`; mutation requests under `/api/admin/*` return
`405 admin_api_read_only`.
