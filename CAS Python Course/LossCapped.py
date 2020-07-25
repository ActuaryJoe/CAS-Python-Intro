def limited_loss(loss, limit):
  """
  This function assumes that losses are from-ground-up
  """
  loss = min(loss, limit)
  return loss