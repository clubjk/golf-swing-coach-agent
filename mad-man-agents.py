import os
from openai import OpenAI
from colorama import init, Fore, Style

init(autoreset=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===================================================================
# SYSTEM PROMPTS — Faithful to the characters
# ===================================================================
DON_SYSTEM = """You are Donald Francis Draper, Creative Director at Sterling Cooper Draper Pryce.
You are brilliant, elegant, deeply cynical, and poetic. You speak in short, powerful sentences with 1960s swagger.
You sell feelings, not products. You see the human truth behind every desire.

When Peggy presents a slogan:
• If it is truly great — praise her lavishly but coolly. Tell her this is the one. End your response with the single word APPROVED on its own line.
• If it is not great — be sharp, honest, and precise. Tell her exactly why it fails (too literal, generic, clever-for-clever’s-sake, no emotional truth, etc.).
• Always give her clear, actionable direction on how to make it better. Push her to go deeper.
Never break character. Never be polite just to be nice."""

PEGGY_SYSTEM = """You are Peggy Olson, junior copywriter at Sterling Cooper Draper Pryce.
You are ambitious, talented, and still proving yourself. You respect Don deeply and learn fast.
Present each slogan clearly, followed by a short, confident rationale (1-2 sentences max).
When Don gives feedback, acknowledge it respectfully, then come back stronger with an improved version based exactly on his notes.
Stay completely in character."""

# ===================================================================
# AGENT HELPER
# ===================================================================
def get_response(system_prompt, messages):
    completion = client.chat.completions.create(
        model="gpt-4o",          # or gpt-4o-mini for cheaper testing
        messages=[{"role": "system", "content": system_prompt}] + messages,
        temperature=0.78,
        max_tokens=450
    )
    return completion.choices[0].message.content.strip()

# ===================================================================
# MAIN SCRIPT
# ===================================================================
print(Fore.CYAN + Style.BRIGHT + "\n" + "="*70)
print("     STERLING COOPER DRAPER PRYCE — CREATIVE MEETING ROOM")
print("="*70 + Style.RESET_ALL)

brief = input(Fore.WHITE + "Campaign brief / product: " + Style.RESET_ALL).strip()
if not brief:
    brief = "Sunkist oranges and soda"

messages = [{"role": "user", "content": f"Brief: {brief}\n\nPeggy, bring me your best slogan."}]

round_num = 1
max_rounds = 8

while round_num <= max_rounds:
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}— ROUND {round_num} —{Style.RESET_ALL}")

    # Peggy presents
    print(Fore.MAGENTA + "Peggy Olson:" + Style.RESET_ALL)
    peggy_reply = get_response(PEGGY_SYSTEM, messages)
    print(peggy_reply)
    messages.append({"role": "assistant", "content": peggy_reply})

    # Don reviews
    print(Fore.RED + Style.BRIGHT + "Don Draper:" + Style.RESET_ALL)
    don_reply = get_response(DON_SYSTEM, messages)
    print(don_reply)
    messages.append({"role": "assistant", "content": don_reply})

    # Check for approval
    if "APPROVED" in don_reply:
        print(Fore.GREEN + Style.BRIGHT + "\n✅ DON APPROVED THE SLOGAN!")
        print(Style.BRIGHT + "\nFINAL APPROVED LINE:")
        # Extract the last Peggy line as the winner
        last_peggy = next(m["content"] for m in reversed(messages) if m["role"] == "assistant" and "Slogan:" in m["content"] or len(m["content"]) < 80)
        print(Fore.YELLOW + last_peggy + Style.RESET_ALL)
        break

    round_num += 1

    if round_num > max_rounds:
        print(Fore.RED + "\nMax rounds reached. Don is still waiting for greatness.")
        break

    input(Fore.WHITE + "\nPress Enter for Peggy to try again..." + Style.RESET_ALL)

print(Fore.CYAN + Style.BRIGHT + "\nMeeting adjourned." + Style.RESET_ALL)