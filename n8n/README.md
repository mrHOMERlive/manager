# n8n Workflow Templates for SPRUT 3.0

## Overview

This directory contains n8n workflow JSON templates that handle Telegram integration,
Google Drive sync, and Obsidian vault synchronization.

## Workflows

| File | Purpose | Status |
|------|---------|--------|
| `workflows/telegram_main.json` | Telegram webhook to SPRUT API pipeline | Ready to configure |
| `workflows/gdrive_sync.json` | Google Drive file sync | Placeholder |
| `workflows/obsidian_sync.json` | Obsidian vault sync via Google Drive | Placeholder |

## Import Instructions

### Prerequisites

1. n8n must be running (part of `docker-compose up`)
2. Access n8n UI at `http://localhost:5678`
3. Have your Telegram Bot Token ready

### Step 1: Configure Credentials in n8n

Before importing workflows, set up the required credentials in n8n:

1. **Telegram Bot API**
   - Go to **Settings > Credentials > Add Credential**
   - Select **Telegram API**
   - Enter your Bot Token (from @BotFather)
   - Save as `SPRUT Telegram Bot`

2. **SPRUT API Bearer Auth**
   - Go to **Settings > Credentials > Add Credential**
   - Select **Header Auth**
   - Name: `Authorization`
   - Value: `Bearer <your_API_SECRET_KEY_from_.env>`
   - Save as `SPRUT API Bearer`

### Step 2: Set Environment Variables in n8n

Add the following environment variable in n8n settings or via Docker:

- `ALLOWED_USER_IDS` - Comma-separated Telegram user IDs (e.g., `123456789,987654321`)

This is already configured in `docker-compose.yml` if the `.env` file contains the variable.

### Step 3: Import Workflows

1. Open n8n UI at `http://localhost:5678`
2. Click **Add workflow** (+ button)
3. Click the **...** menu in the top-right corner
4. Select **Import from File...**
5. Choose the desired `.json` file from this directory
6. Repeat for each workflow

### Step 4: Activate Workflows

1. Open the imported **SPRUT Telegram Main** workflow
2. Update credential references if names differ from defaults
3. Toggle the workflow to **Active** (top-right switch)
4. Test by sending a message to your Telegram bot

## Telegram Main Workflow Flow

```
Telegram Trigger (webhook)
    |
    v
Security Check (Code node)
  - Validates user_id against ALLOWED_USER_IDS
  - Extracts message type (text/voice/photo/document)
  - Rejects unauthorized users
    |
    v
SPRUT API Process (HTTP Request)
  - POST http://sprut-api:8000/api/process
  - Bearer token authentication
  - Sends { message, user_id } payload
    |
    v
Check Voice Data (Switch)
  - If voice_data present -> Send Text + Send Voice
  - If no voice_data     -> Send Text only
    |               |
    v               v
Send Text +     Send Text
Send Voice      Message
```

## Troubleshooting

- **Webhook not receiving messages:** Ensure the Telegram bot webhook is set.
  n8n sets this automatically when the workflow is activated.
- **401 Unauthorized from SPRUT API:** Check the Bearer token in the Header Auth
  credential matches `API_SECRET_KEY` in `.env`.
- **User rejected:** Verify the Telegram user ID is in `ALLOWED_USER_IDS`.
- **Timeout errors:** The SPRUT API timeout is set to 120 seconds. For long
  AI processing, this may need to be increased in the HTTP Request node settings.
