if [[ -r "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh" ]]; then
  source "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh"
fi

source ~/powerlevel10k/powerlevel10k.zsh-theme

[[ ! -f ~/.p10k.zsh ]] || source ~/.p10k.zsh
source $HOME/.nvm/nvm.sh

source /root/.oh-my-zsh/custom/plugins/zsh-autosuggestions/zsh-autosuggestions.zsh

alias att='source ~/.zshrc'
alias config='code $HOME/.zshrc'
alias pypath='source /product/pypath.sh'

HISTSIZE=10000
SAVEHIST=10000
HISTFILE=~/.zsh_history
setopt HIST_IGNORE_ALL_DUPS
setopt HIST_IGNORE_SPACE
setopt INC_APPEND_HISTORY
setopt SHARE_HISTORY
setopt EXTENDED_HISTORY


plugins=(
  git
  z
  zsh-autosuggestions
  zsh-syntax-highlighting
)

bindkey "^[[1;5D" backward-word
bindkey "^[[1;5C" forward-word

[ -f ~/.fzf.zsh ] && source ~/.fzf.zsh

export EDITOR=nano
export VISUAL=nano