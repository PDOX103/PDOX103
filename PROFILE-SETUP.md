# PDOX103 GitHub Profile - Setup Notes

This repository is filled with the identity and links found in the uploaded profile:

- Name: Md. Fahmidul Karim Rafi
- GitHub username: PDOX103
- Profile repository: PDOX103/PDOX103
- Degree: Computer Science and Engineering
- University: Ahsanullah University of Science and Technology
- Email: fahmidulkarimrafi2.0@gmail.com
- LinkedIn: md-fahmidul-karim-rafi
- Facebook: fahmidulkarim.paradox.103
- Instagram: fahmidulkarim
- YouTube: @RaFi-cf8cn

## Upload these files

Upload the following to the root of the public `PDOX103/PDOX103` repository:

- `README.md`
- `dark.svg`
- `light.svg`

Upload this workflow at:

- `.github/workflows/snake.yml`

## Enable the contribution snake

Inside the `PDOX103/PDOX103` repository:

1. Open **Settings**.
2. Open **Actions > General**.
3. Scroll to **Workflow permissions**.
4. Select **Read and write permissions**.
5. Save.
6. Open **Actions > Generate Snake Animation**.
7. Select **Run workflow**.
8. Wait until the run is green and the `output` branch exists.

The README snake will appear only after the output branch has been created.

## Self-host the statistics cards

The README currently uses temporary public statistics endpoints so the profile renders immediately. Public endpoints can return rate-limit errors.

For the PDF-recommended setup:

1. Create a GitHub classic token.
2. Give it the `repo` scope.
3. Copy it immediately and treat it as a password.
4. Never paste the token into chat, a public repository, or a website form.
5. Fork `anuraghazra/github-readme-stats`.
6. Import the fork into Vercel using the Hobby plan.
7. Add the environment variable `PAT_1` and use the token as its value.
8. Deploy.
9. In `README.md`, replace both occurrences of:

   `https://github-readme-stats.vercel.app`

   with:

   `https://YOUR-VERCEL-INSTANCE.vercel.app`

## Banner limitation

The supplied repository did not contain a portrait photo or logo reference files. The included `dark.svg` and `light.svg` are therefore clean terminal-style banners using the correct profile information, not the photo-dithered morphing banner from the Master Prompt.

To build the portrait version accurately, supply:

- One sharp head-and-shoulders photo
- A flat, uniform background
- Even face lighting
- At least 1000 px on the short edge
- Three real logo reference images

Do not fabricate a portrait from another person's photo.
