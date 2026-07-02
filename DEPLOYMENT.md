# Deploying to Azure (Portal + GitHub, no CLI)

> **Note**: this file is listed in `.gitignore` on purpose and will never be
> pushed to your GitHub repo -- it's local-only reference material for you,
> not part of the submitted project.

This guide deploys the SHL Assessment Recommender to Azure App Service,
using your Azure for Students account, driven entirely from the GitHub
website + Azure Portal. The only unavoidable terminal use is pushing your
code to GitHub for the first time -- and even that can be done with zero
terminal via GitHub Desktop (covered below).

## What we're building, and why

| Resource | What it is | Why we need it |
|---|---|---|
| **Resource Group** | A folder that groups related Azure resources together | Lets you see/manage/delete everything for this project as one unit |
| **App Service Plan (B1)** | The virtual machine capacity your app runs on | Hosts the actual compute; B1 gives 10GB disk (needed for the ~1.2GB of ML dependencies) and ~$13/month prorated cost, covered by your student credit |
| **App Service (Web App)** | The actual hosted instance of your FastAPI app | This is what serves `/health` and `/chat` at a public HTTPS URL |
| ~~Database~~ | *(not used)* | Your catalog is static data committed as files in the repo (`data/catalog.json`, `data/vectorstore/`). There's no live read/write data that needs a database -- adding one would just add cost and a moving part with no benefit. |

You will **not** need: Azure CLI, PowerShell scripts, Azure Storage, or a
database service.

---

## Part 1 -- Get your code onto GitHub

### Option A: GitHub Desktop (fully GUI, recommended if you want zero terminal)

1. Install [GitHub Desktop](https://desktop.github.com/) and sign in with
   your GitHub account.
2. **File -> Add local repository** -> browse to
   `C:\Users\Urvi\Downloads\SHL_AI_Intern_Assignment` -> if it says "this
   directory is not a git repository, create one?", click **create a
   repository**.
3. Fill in a summary like "Initial commit" and click **Commit to main**.
4. Click **Publish repository** (top bar). Name it (e.g.
   `shl-assessment-recommender`), leave **Keep this code private** checked
   or uncheck it -- your choice -- and click **Publish Repository**.
5. Done -- your code is now on GitHub. Every time you want to update the
   deployed app later, make changes locally, then in GitHub Desktop:
   **Commit to main -> Push origin**. That's the only "workflow" you'll ever
   need to touch again.

### Option B: minimal git commands (if you'd rather not install GitHub Desktop)

This is the one place a terminal is genuinely unavoidable -- there's no
portal-only way to push a first commit. Four commands, run once, from the
project folder:

```powershell
git init
git add .
git commit -m "Initial commit"
git branch -M main
```

Then create an empty repository on [github.com/new](https://github.com/new)
(name it, e.g., `shl-assessment-recommender`, **do not** check "Add a
README" since you already have one), copy the URL it gives you, and run:

```powershell
git remote add origin https://github.com/<your-username>/shl-assessment-recommender.git
git push -u origin main
```

> **Before pushing**: double check `.env` is NOT in the repo (it's already
> in `.gitignore`, so `git status` should never show it). Your real API key
> stays local; you'll enter it again through the Azure Portal in Part 3.

---

## Part 2 -- Create the Web App in the Azure Portal

1. Go to [portal.azure.com](https://portal.azure.com) and sign in with your
   student account.
2. Click **Create a resource** (top-left, `+` icon).

   > 📸 *[Placeholder: Azure Portal home page, red arrow pointing at the
   > "+ Create a resource" button in the top-left corner]*

3. Search for **"Web App"** and click **Create**.
4. Fill in the **Basics** tab:

   | Field | Value | Notes |
   |---|---|---|
   | Subscription | *Azure for Students* | Make sure this shows your student subscription, not "Free Trial" or another one |
   | Resource Group | Click **Create new** -> `shl-recommender-rg` | Groups all resources for this project |
   | Name | `shl-recommender-<yourname>` | Must be globally unique across all of Azure -- this becomes `https://<name>.azurewebsites.net` |
   | Publish | **Code** | We're deploying source code, not a container |
   | Runtime stack | **Python 3.11** | Matches what this project was built/tested against |
   | Operating System | **Linux** | Required for Python on App Service |
   | Region | Pick one near you, e.g. *Central India* | Lower latency |

   > 📸 *[Placeholder: "Create Web App" Basics tab, with Subscription,
   > Resource Group, Name, Publish, Runtime stack, OS, and Region fields
   > visible and filled in as above]*

5. Click **Next: Deployment ->**. Leave "Continuous deployment" **off** for
   now -- we'll set that up properly in Part 3 with the full GitHub Actions
   wizard (doing it here skips some options).
6. Click **Next: Pricing plans ->**. Click **Create new** under App Service
   Plan if one isn't already selected, name it `shl-recommender-plan`, and
   under **Sku and size** click **Change size** -> select the **Dev/Test**
   tab -> choose **B1**.

   > 📸 *[Placeholder: Pricing plan "Spec Picker" dialog, Dev/Test tab
   > selected, B1 tier highlighted, showing "~$12.41/month" estimated cost]*

   This is the tier choice that matters most: **F1 (Free) will not work**
   for this app because it only allows 1GB of disk, and our ML dependencies
   (torch, sentence-transformers) need about 1.2GB installed. B1 gives 10GB
   and costs a few cents a day, which your $100 credit absorbs easily.

7. Click **Review + create**, wait for validation to pass, then **Create**.
8. Wait for deployment to finish (~1 minute), then click **Go to resource**.

You now have an empty Web App -- it's live at your `.azurewebsites.net` URL
but has no code deployed yet.

---

## Part 3 -- Connect GitHub for automatic deployment (CI/CD)

This is the step that makes "push to `main` -> auto redeploy" work, entirely
configured by clicking through the Portal (Azure writes the GitHub Actions
workflow file for you).

1. In your Web App's left-hand menu, under **Deployment**, click
   **Deployment Center**.

   > 📸 *[Placeholder: Web App left sidebar, "Deployment Center" highlighted
   > under the "Deployment" section]*

2. Under **Source**, choose **GitHub**.
3. Click **Authorize** and sign in to GitHub if prompted -- this lets Azure
   install a small GitHub App on your account/repo so it can push workflow
   files and read your code.
4. Fill in:
   - **Organization**: your GitHub username
   - **Repository**: `shl-assessment-recommender` (or whatever you named it)
   - **Branch**: `main`
5. **Build provider**: leave as **GitHub Actions** (the default).

   > 📸 *[Placeholder: Deployment Center form with Source=GitHub,
   > Organization/Repository/Branch dropdowns filled in, Build provider =
   > GitHub Actions]*

6. Click **Save**.

What just happened: Azure committed a new file,
`.github/workflows/main_<your-app-name>.yml`, directly into your GitHub
repo. It also securely stored a deployment credential as a GitHub Actions
secret on your repo (you never see or copy this manually). From now on,
**every `git push` to `main` triggers that workflow**, which installs your
`requirements.txt` dependencies and deploys the result to your Web App
automatically.

7. Click the **Logs** tab in Deployment Center to watch the first deployment
   run (typically 3-6 minutes, mostly spent installing torch). You can also
   watch it from the GitHub side: go to your repo on github.com -> **Actions**
   tab -> click the running workflow.

   > 📸 *[Placeholder: Deployment Center "Logs" tab showing a deployment run
   > in progress with a green checkmark once complete]*

---

## Part 4 -- Configure environment variables (your API key etc.)

Your `.env` file never left your computer (it's gitignored) -- you set the
same values here instead, through the Portal.

1. In your Web App's left-hand menu, under **Settings**, click
   **Environment variables** (older Portal versions call this
   **Configuration**).

   > 📸 *[Placeholder: Web App left sidebar, "Environment variables"
   > highlighted under "Settings"]*

2. Under the **App settings** tab, click **+ Add** for each of the
   following (name / value pairs):

   | Name | Value |
   |---|---|
   | `LLM_PROVIDER` | `groq` |
   | `LLM_API_KEY` | *your real Groq key, `gsk_...`* |
   | `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` |
   | `LOG_LEVEL` | `INFO` |
   | `SCM_DO_BUILD_DURING_DEPLOYMENT` | `true` |

   > 📸 *[Placeholder: App settings tab with the 5 name/value rows above
   > entered, "+ Add" button visible above the list]*

   The last one (`SCM_DO_BUILD_DURING_DEPLOYMENT`) tells Azure to actually
   run `pip install -r requirements.txt` after receiving your code, rather
   than expecting a pre-built environment.

3. Still on this page, click the **General settings** tab (same
   Environment variables / Configuration screen) and find **Startup
   Command**. Enter exactly:
   ```
   uvicorn app.api:app --host 0.0.0.0 --port 8000
   ```
   This tells Azure how to actually launch your app -- FastAPI/uvicorn has
   no Azure-recognized default the way Django does, so this is required.

   > 📸 *[Placeholder: General settings tab, "Startup Command" text field
   > with the uvicorn command entered]*

4. Click **Save** at the top, then click **Continue** when it warns the app
   will restart.

---

## Part 5 -- Verify it's working (all in the browser, no terminal)

1. Go to your Web App's **Overview** page and copy the **Default domain**
   (looks like `https://shl-recommender-yourname.azurewebsites.net`).
2. Open a new browser tab and go to `<that-url>/health`. You should see:
   ```json
   {"status":"ok"}
   ```
   First load after a deploy or period of inactivity can take up to a
   couple of minutes while the app container starts and the embedding model
   loads -- this is expected and mentioned in the assignment's own spec.
3. Go to `<that-url>/docs` -- this is the interactive Swagger UI (built into
   FastAPI, no setup needed). Click **POST /chat -> Try it out**, paste:
   ```json
   {"messages": [{"role": "user", "content": "Hiring a Java developer who works with stakeholders"}]}
   ```
   and click **Execute**. You should get a real reply back (e.g. asking
   about seniority), confirming the whole pipeline -- FastAPI, the retriever,
   FAISS index, and your Groq API key -- is working end to end.

   > 📸 *[Placeholder: Swagger UI at /docs, POST /chat expanded, "Try it
   > out" clicked, request body filled in, showing a 200 response below]*

---

## Ongoing workflow (after today)

Any time you change code:
1. Edit locally as usual.
2. GitHub Desktop: **Commit to main -> Push origin** (or `git add . && git commit -m "..." && git push` if using the CLI path).
3. That's it -- GitHub Actions rebuilds and redeploys automatically. Watch
   progress under your repo's **Actions** tab on github.com.

## Cleanup (stop billing when you're done)

Go to the **`shl-recommender-rg`** resource group in the Portal -> click
**Delete resource group** -> type the resource group name to confirm. This
removes the Web App and App Service Plan together in one action -- nothing
else in your subscription is affected.

## Submission

Submit `https://<your-app-name>.azurewebsites.net` as your public API
endpoint URL -- both `/health` and `/chat` are reachable at that base.
